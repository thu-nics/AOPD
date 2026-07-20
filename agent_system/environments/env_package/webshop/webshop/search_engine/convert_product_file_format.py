import argparse
from pathlib import Path
import sys
import json
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from web_agent_site.utils import DEFAULT_FILE_PATH, DEFAULT_ATTR_PATH
from web_agent_site.engine.engine import load_products

parser = argparse.ArgumentParser(description='Convert WebShop products to Pyserini JSONL')
parser.add_argument('--file-path', type=Path, default=Path(DEFAULT_FILE_PATH))
parser.add_argument('--attr-path', type=Path, default=Path(DEFAULT_ATTR_PATH))
parser.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()

all_products, *_ = load_products(filepath=args.file_path, attrpath=args.attr_path)


docs = []
for p in tqdm(all_products, total=len(all_products)):
    option_texts = []
    options = p.get('options', {})
    for option_name, option_contents in options.items():
        option_contents_text = ', '.join(option_contents)
        option_texts.append(f'{option_name}: {option_contents_text}')
    option_text = ', and '.join(option_texts)

    doc = dict()
    doc['id'] = p['asin']
    doc['contents'] = ' '.join([
        p['Title'],
        p['Description'],
        p['BulletPoints'][0],
        option_text,
    ]).lower()
    doc['product'] = p
    docs.append(doc)


def write_documents(directory, documents):
    output_dir = args.output_root / directory
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "documents.jsonl").open("w") as output:
        for doc in documents:
            output.write(json.dumps(doc) + "\n")


write_documents("resources_100", docs[:100])
write_documents("resources", docs)
write_documents("resources_1k", docs[:1000])
write_documents("resources_100k", docs[:100000])
