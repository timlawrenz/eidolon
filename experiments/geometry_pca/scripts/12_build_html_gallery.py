#!/usr/bin/env python3
"""
Generates a simple HTML gallery to quickly scroll and review the 120 identity collages.
Outputs to output/collages_120/review.html.
"""
import os, glob

OUT_DIR = "output/collages_120"

def main():
    collages = sorted(glob.glob(os.path.join(OUT_DIR, "*.png")))
    
    html = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "<style>",
        "body { background-color: #1a1a1a; color: #fff; font-family: sans-serif; margin: 20px; }",
        ".collage-container { margin-bottom: 40px; border-bottom: 1px solid #444; padding-bottom: 20px; }",
        "img { max-width: 100%; height: auto; border: 1px solid #333; }",
        "h2 { color: #4CAF50; }",
        ".btn { background-color: #f44336; color: white; border: none; padding: 5px 10px; cursor: pointer; margin-left: 20px; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Identity Verification Gallery</h1>",
        "<p>Review each identity below. If you spot a male face, a non-face, or merged identities, note the name.</p>"
    ]
    
    for c in collages:
        name = os.path.basename(c).replace(".png", "")
        html.append(f"<div class='collage-container'>")
        html.append(f"<h2>{name}</h2>")
        # Use relative paths since HTML is in the same dir
        html.append(f"<img src='{os.path.basename(c)}' loading='lazy'>")
        html.append(f"</div>")
        
    html.append("</body></html>")
    
    out_path = os.path.join(OUT_DIR, "review.html")
    with open(out_path, "w") as f:
        f.write("\n".join(html))
        
    print(f"Gallery generated at: {out_path}")

if __name__ == "__main__":
    main()
