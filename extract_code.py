
import json

notebook_path = 'social-media-addiction-data-analysis-modeling.ipynb'
output_path = 'social_media_analysis.py'

try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    with open(output_path, 'w', encoding='utf-8') as f:
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                source = cell['source']
                if isinstance(source, list):
                    source = "".join(source)
                f.write(source + "\n\n")
    print(f"Successfully extracted code to {output_path}")

except Exception as e:
    print(f"Error extracting code: {e}")
