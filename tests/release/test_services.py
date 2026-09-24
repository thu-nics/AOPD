import os
import socket
import subprocess
import sys
import time

import pytest

from aopd.services import local_command, stop_owned


def test_port_probe_allows_closed_server_time_wait():
    from aopd.services import require_free_port

    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
        server.listen()
        with socket.create_connection(("127.0.0.1", port)) as client:
            connection, _ = server.accept()
            connection.close()
            assert client.recv(1) == b""
    require_free_port(port)


def test_port_probe_rejects_listening_server():
    from aopd.services import require_free_port

    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen()
        with pytest.raises(OSError):
            require_free_port(server.getsockname()[1])


@pytest.mark.parametrize("override", ["--model=/other", "--host", "--served-model-name", "--config", "-tp"])
def test_server_extra_args_cannot_override_owned_identity(tmp_path, override):
    service = {"model_path": "/model", "model": "served", "gpus": [0], "port": 8100, "extra_args": [override]}
    with pytest.raises(ValueError, match="extra_args"):
        local_command("user", service, tmp_path)


def test_local_service_keeps_physical_devices_and_separate_python(tmp_path):
    command, env = local_command(
        "aux",
        {
            "python": "/venv/aux/python",
            "model_path": "/models/qwen",
            "model": "qwen",
            "gpus": [2, 6],
            "tp": 2,
            "dp": 1,
            "port": 8123,
        },
        tmp_path,
    )
    assert command[0] == "/venv/aux/python"
    assert env["CUDA_VISIBLE_DEVICES"] == "2,6"
    assert command[command.index("--tensor-parallel-size") + 1] == "2"
    assert command[command.index("--host") + 1] == "127.0.0.1"


def test_cleanup_stops_only_process_it_owns():
    owned = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
    try:
        stop_owned(owned, timeout=2)
        assert owned.poll() is not None
        assert unrelated.poll() is None
    finally:
        stop_owned(owned, timeout=2)
        unrelated.terminate()
        unrelated.wait(timeout=2)


def test_unknown_model_cli_arguments_are_not_shell_expanded(tmp_path):
    with pytest.raises(ValueError, match="extra_args"):
        local_command("aux", {"model_path": "/models/qwen", "model": "qwen", "gpus": [0], "port": 8123, "extra_args": "--x $(bad)"}, tmp_path)


def test_cleanup_reaps_remaining_group_after_leader_exits():
    leader = subprocess.Popen([sys.executable, "-c", "import os,time,signal; p=os.fork(); print(p,flush=True) if p else None; os._exit(0) if p else None; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(120)"], stdout=subprocess.PIPE, text=True, start_new_session=True)
    child = int(leader.stdout.readline())
    leader.wait(timeout=2)
    time.sleep(0.1)
    try:
        stop_owned(leader, timeout=0.2)
        import psutil

        for _ in range(30):
            if not psutil.pid_exists(child) or psutil.Process(child).status() == psutil.STATUS_ZOMBIE:
                break
            time.sleep(0.02)
        else:
            pytest.fail("owned child survived cleanup")
    finally:
        try:
            os.kill(child, 9)
        except ProcessLookupError:
            pass
