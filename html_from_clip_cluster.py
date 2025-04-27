import os
import json

def create_html_from_clusters_json(json_path, output_html_path):
    """
    Reads clustering result from JSON and creates an HTML file to view clusters
    """
    # Load clusters from JSON
    with open(json_path, "r") as f:
        clusters_data = json.load(f)

    clusters = clusters_data["clusters"]

    html = """
    <html>
    <head>
        <title>Clustered Images</title>
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            .cluster { margin-bottom: 50px; }
            .cluster-title { font-size: 24px; font-weight: bold; margin-bottom: 10px; }
            .image-row { display: flex; flex-wrap: wrap; gap: 10px; }
            img { max-width: 200px; max-height: 200px; border: 1px solid #ccc; padding: 5px; }
        </style>
    </head>
    <body>
        <h1>Clustered Images Viewer</h1>
    """

    for cluster_label, items in clusters.items():
        html += f'<div class="cluster">\n'
        html += f'<div class="cluster-title">Cluster {cluster_label} ({len(items)} images)</div>\n'
        html += f'<div class="image-row">\n'
        for item in items:
            img_path = item["filepath"]
            img_path = img_path.replace(" ", "%20")  # URL encode spaces
            html += f'<img src="file://{img_path}" alt="Image">\n'
        html += '</div>\n</div>\n'

    html += """
    </body>
    </html>
    """

    with open(output_html_path, "w") as f:
        f.write(html)

    print(f"✅ HTML file created: {output_html_path}")

# Example usage:
if __name__ == "__main__":
    create_html_from_clusters_json(
        # json_path="/Users/sumit/mmv-python/clip_embeddings/vectors_clusters.json",  # ← your saved clustering .json
        json_path="/Users/sumit/mmv-python/clip_embeddings/vectors_clusters_hdbscan.json",
        output_html_path="/Users/sumit/Desktop/clusters_viewer.html"                      # ← where you want the HTML
    )
