#!/usr/bin/env python3
"""
Generate a standalone static HTML report by embedding results.json directly into index.html.
This creates a self-contained file that can be opened directly in a browser without any server.
"""

import base64
import json
import sys
from pathlib import Path

def create_static_report():
    """Generate static HTML report with embedded data."""
    script_dir = Path(__file__).parent

    # Read the template HTML
    index_path = script_dir / 'index.html'
    if not index_path.exists():
        print(f"❌ Error: {index_path} not found")
        sys.exit(1)

    with open(index_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Read the results JSON
    results_path = script_dir / 'results.json'
    if not results_path.exists():
        print(f"❌ Error: {results_path} not found")
        print("Please run generate_results.py first to create results.json")
        sys.exit(1)

    with open(results_path, 'r', encoding='utf-8') as f:
        results_data = json.load(f)

    # Convert JSON to JavaScript variable
    # Using JSON.stringify-like formatting for clean output
    embedded_data = json.dumps(results_data, indent=2)

    # Replace the fetch() logic with embedded data using an IIFE pattern.
    # Instead of hardcoding function calls (which fall out of sync), we
    # preserve the .then() callback body and invoke it directly with
    # embedded data. This way any new display functions added to index.html
    # are automatically included in the static version.

    # Step 1: Replace fetch + response.json chain with an IIFE
    fetch_chain = (
        "fetch('results.json')\n"
        "  .then(response => response.json())\n"
        "  .then(data => {"
    )
    iife_open = (
        f"const __embeddedResults = {embedded_data};\n"
        "((data) => {"
    )

    if fetch_chain not in html_content:
        print("❌ Error: Could not find fetch() chain in HTML")
        sys.exit(1)

    static_html = html_content.replace(fetch_chain, iife_open, 1)

    # Step 2: Replace the .catch() block with the IIFE closing invocation
    catch_block = (
        "  })\n"
        "  .catch(error => {\n"
        "    console.error('Error loading results:', error);\n"
        "    document.getElementById('loading-metrics').innerHTML =\n"
        "      `<p style=\"color: #c62828;\">❌ Error loading results. "
        "Please run <code>python generate_results.py</code> first.</p>`;\n"
        "  });"
    )
    iife_close = "  })(__embeddedResults);"

    if catch_block not in static_html:
        print("❌ Error: Could not find .catch() block in HTML")
        sys.exit(1)

    static_html = static_html.replace(catch_block, iife_close, 1)

    # Update title to indicate this is the static version
    static_html = static_html.replace(
        '<title>FAMAIL: Trajectory Modification Results (Pseudo-Causal Fairness)</title>',
        '<title>FAMAIL: Trajectory Modification Results (Pseudo-Causal Fairness) - Static Report</title>'
    )

    static_html = static_html.replace(
        '<div class="subtitle">Progress Report: Pseudo-Causal Fairness Formulation</div>',
        '<div class="subtitle">Progress Report: Pseudo-Causal Fairness Formulation (Static Report)</div>'
    )

    # Embed images as base64 data URIs so the file is fully self-contained
    assets_dir = script_dir.parent / 'assets'
    image_replacements = {
        '../assets/spatial_mod_result_L-before_R-after.png': assets_dir / 'spatial_mod_result_L-before_R-after.png',
        '../assets/causal_mod_result_L-before_R-after.png': assets_dir / 'causal_mod_result_L-before_R-after.png',
    }

    for rel_path, abs_path in image_replacements.items():
        if abs_path.exists():
            with open(abs_path, 'rb') as img_file:
                img_b64 = base64.b64encode(img_file.read()).decode('utf-8')
            data_uri = f'data:image/png;base64,{img_b64}'
            static_html = static_html.replace(f'src="{rel_path}"', f'src="{data_uri}"')
            print(f"  ✅ Embedded: {abs_path.name} ({abs_path.stat().st_size / 1024:.0f} KB)")
        else:
            print(f"  ⚠️  Image not found (skipping): {abs_path}")

    # Write the static report
    output_path = script_dir / 'report_static.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(static_html)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)

    print("✅ Static report created successfully!")
    print(f"\n📄 Output file: {output_path}")
    print(f"📊 File size: {file_size_mb:.2f} MB")
    print(f"\n🎯 Usage:")
    print(f"   1. Send {output_path.name} to your boss")
    print(f"   2. She can double-click it to open in any browser")
    print(f"   3. No server, no JSON file, no setup required!")
    print(f"\n✨ All features preserved:")
    print(f"   • Interactive dropdowns")
    print(f"   • Plotly visualizations")
    print(f"   • Math rendering (KaTeX)")
    print(f"   • Full precision iteration details")
    print(f"   • Embedded trajectory visualization images")

if __name__ == '__main__':
    create_static_report()
