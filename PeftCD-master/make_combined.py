import os

base_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(base_dir, 'Combined')
os.makedirs(out_dir, exist_ok=True)

datasets = {
    'SYSU-CD': os.path.join(base_dir, 'SYSU-CD'),
    'LEVIR-CD+': os.path.join(base_dir, 'LEVIR-CD+'),
}

for split in ['train', 'val', 'test']:
    combined_lines = []
    for name, dpath in datasets.items():
        txt_path = os.path.join(dpath, f'{split}.txt')
        if not os.path.exists(txt_path):
            print(f"Skipping {txt_path} — not found")
            continue
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split('  ')
                abs_parts = []
                for p in parts:
                    if not os.path.isabs(p):
                        abs_parts.append(os.path.join(dpath, p))
                    else:
                        abs_parts.append(p)
                combined_lines.append('  '.join(abs_parts))
        print(f"{name}/{split}: {len(open(txt_path).readlines())} samples")

    out_path = os.path.join(out_dir, f'{split}.txt')
    with open(out_path, 'w') as f:
        f.write('\n'.join(combined_lines) + '\n')
    print(f"Combined {split}: {len(combined_lines)} samples → {out_path}\n")
