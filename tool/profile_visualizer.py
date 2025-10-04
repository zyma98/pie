#!/usr/bin/env python3
"""
Simple web-based profiling visualization tool.

Usage:
    # Start with an initial file
    python tool/profile_visualizer.py <path_to_profiling_json>

    # Start without a file (uses current directory for file search)
    python tool/profile_visualizer.py

    Example:
    python tool/profile_visualizer.py pie-metal/20251003_205526_profiling_result.json
    python tool/profile_visualizer.py

Features:
    - Load files from server directory dropdown
    - Drag & drop JSON files
    - Click to browse local files
"""

import json
import sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
import os


class ProfileVisualizerHandler(SimpleHTTPRequestHandler):
    """Custom handler to serve profiling visualization."""

    profiling_data = None
    search_directory = None
    root_directory = None  # The initial search directory - can't go above this

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(self.generate_html().encode())
        elif self.path == "/data.json":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.profiling_data).encode())
        elif self.path.startswith("/list_files"):
            # List JSON files in the search directory and subdirectories
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            base_path = query.get('path', [''])[0]
            if not base_path:
                base_path = self.search_directory
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            files = self._list_json_files(base_path)
            self.wfile.write(json.dumps(files).encode())
        elif self.path.startswith("/list_dirs"):
            # List subdirectories
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            base_path = query.get('path', [''])[0]
            if not base_path:
                base_path = self.search_directory
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            dirs = self._list_directories(base_path)
            self.wfile.write(json.dumps(dirs).encode())
        elif self.path.startswith("/load_file?"):
            # Load a specific file
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            filepath = query.get('path', [''])[0]
            if filepath and os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    self.send_response(200)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(data).encode())
                except Exception as e:
                    self.send_error(400, f"Error loading file: {str(e)}")
            else:
                self.send_error(404, "File not found")
        else:
            self.send_error(404, "File not found")

    def _list_json_files(self, base_path=''):
        """List JSON files in the specified directory (non-recursive)."""
        files = []
        if not base_path:
            base_path = self.search_directory

        print(f"[DEBUG] _list_json_files called with base_path='{base_path}'")

        # No security restriction - allow browsing anywhere
        abs_base = os.path.abspath(base_path)

        if os.path.exists(base_path) and os.path.isdir(base_path):
            try:
                items = os.listdir(base_path)
                for item in items:
                    filepath = os.path.join(base_path, item)
                    if os.path.isfile(filepath) and item.endswith('.json'):
                        stat = os.stat(filepath)
                        files.append({
                            'path': filepath,
                            'name': item,
                            'size': stat.st_size,
                            'mtime': stat.st_mtime,
                            'relative_path': os.path.relpath(filepath, self.root_directory)
                        })
                        print(f"[DEBUG] Added JSON file: {item}")
            except PermissionError as e:
                print(f"[DEBUG] Files: PermissionError: {e}")
                pass

        # Sort by modification time, newest first
        files.sort(key=lambda x: x['mtime'], reverse=True)
        print(f"[DEBUG] Returning {len(files)} JSON files")
        return files

    def _list_directories(self, base_path=''):
        """List subdirectories in the specified directory."""
        dirs = []
        if not base_path:
            base_path = self.search_directory

        print(f"[DEBUG] _list_directories called with base_path='{base_path}'")
        print(f"[DEBUG] search_directory='{self.search_directory}'")
        print(f"[DEBUG] root_directory='{self.root_directory}'")

        # No security restriction - allow browsing anywhere
        abs_base = os.path.abspath(base_path)
        if os.path.exists(base_path) and os.path.isdir(base_path):
            try:
                items = os.listdir(base_path)
                print(f"[DEBUG] Found {len(items)} items in directory")
                for item in items:
                    dirpath = os.path.join(base_path, item)
                    if os.path.isdir(dirpath) and not item.startswith('.'):
                        dirs.append({
                            'path': dirpath,
                            'name': item,
                            'relative_path': os.path.relpath(dirpath, self.root_directory)
                        })
            except PermissionError as e:
                print(f"[DEBUG] PermissionError: {e}")
                pass
        else:
            print(f"[DEBUG] Path does not exist or is not a directory: {base_path}")

        # Sort alphabetically
        dirs.sort(key=lambda x: x['name'])

        # Add parent directory option (always show .. to allow going up)
        parent = os.path.dirname(abs_base)
        if parent != abs_base:  # Not at filesystem root
            dirs.insert(0, {
                'path': parent,
                'name': '..',
                'relative_path': os.path.relpath(parent, self.root_directory) if parent != self.root_directory else '.'
            })
            print(f"[DEBUG] Added parent directory: {parent}")

        print(f"[DEBUG] Returning {len(dirs)} directories")
        return dirs

    def generate_html(self):
        """Generate the HTML visualization page."""
        return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Profiling Visualizer</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px 8px 0 0;
        }

        .header h1 {
            font-size: 28px;
            margin-bottom: 10px;
        }

        .header .meta {
            opacity: 0.9;
            font-size: 14px;
        }

        .file-loader {
            padding: 20px 30px;
            background: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
        }

        .file-loader.compact {
            padding: 15px 30px;
        }

        .file-loader h3 {
            margin-bottom: 15px;
            color: #333;
            font-size: 16px;
        }

        .file-loader.compact h3 {
            margin-bottom: 10px;
            font-size: 14px;
        }

        .file-selector {
            margin-bottom: 20px;
        }

        .directory-browser {
            border: 1px solid #ddd;
            border-radius: 6px;
            background: white;
            margin-bottom: 15px;
            max-height: 300px;
            overflow-y: auto;
            position: relative;
        }

        .directory-browser.loading {
            opacity: 0.6;
            pointer-events: none;
        }

        .loading-spinner {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 40px;
            height: 40px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            display: none;
            z-index: 10;
        }

        .directory-browser.loading .loading-spinner {
            display: block;
        }

        @keyframes spin {
            0% { transform: translate(-50%, -50%) rotate(0deg); }
            100% { transform: translate(-50%, -50%) rotate(360deg); }
        }

        .current-path {
            padding: 10px;
            background: #f8f9fa;
            border-bottom: 1px solid #ddd;
            font-size: 13px;
            color: #666;
            font-family: monospace;
        }

        .file-list, .dir-list {
            list-style: none;
            margin: 0;
            padding: 0;
        }

        .file-item, .dir-item {
            padding: 10px 15px;
            border-bottom: 1px solid #f0f0f0;
            cursor: pointer;
            transition: background 0.2s;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .file-item:hover, .dir-item:hover {
            background: #f8f9fa;
        }

        .file-item.selected {
            background: #e8eaff;
        }

        .file-icon, .dir-icon {
            width: 20px;
            text-align: center;
        }

        .file-info {
            flex: 1;
        }

        .file-name {
            font-weight: 500;
            color: #333;
        }

        .dir-name {
            font-weight: 500;
            color: #667eea;
        }

        .file-meta {
            font-size: 12px;
            color: #999;
            margin-top: 2px;
        }

        .load-button {
            padding: 10px 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            transition: transform 0.2s;
            width: 100%;
        }

        .load-button:hover:not(:disabled) {
            transform: translateY(-2px);
        }

        .load-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .separator {
            text-align: center;
            margin: 20px 0;
            color: #999;
            font-size: 13px;
        }

        .drop-zone {
            border: 2px dashed #667eea;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            background: #f8f9ff;
            cursor: pointer;
            transition: all 0.3s;
        }

        .file-loader.compact .drop-zone {
            padding: 20px;
        }

        .drop-zone:hover {
            background: #f0f2ff;
            border-color: #764ba2;
        }

        .drop-zone.drag-over {
            background: #e8eaff;
            border-color: #764ba2;
            border-width: 3px;
        }

        .drop-zone-text {
            color: #667eea;
            font-size: 16px;
            font-weight: 500;
        }

        .file-loader.compact .drop-zone-text {
            font-size: 14px;
        }

        .drop-zone-hint {
            color: #999;
            font-size: 13px;
            margin-top: 8px;
        }

        .file-loader.compact .drop-zone-hint {
            font-size: 12px;
            margin-top: 4px;
        }

        .content-wrapper {
            display: none;
        }

        .content-wrapper.visible {
            display: block;
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
        }

        .stat-card {
            background: white;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #667eea;
        }

        .stat-card .label {
            color: #666;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }

        .stat-card .value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }

        .stat-card .unit {
            font-size: 14px;
            color: #999;
            margin-left: 4px;
        }

        .controls {
            padding: 20px 30px;
            background: white;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            gap: 15px;
            align-items: center;
        }

        .controls label {
            font-size: 14px;
            color: #666;
        }

        .controls select, .controls input {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }

        .tree-container {
            padding: 30px;
            overflow-x: auto;
        }

        .tree-node {
            margin: 0;
            padding: 0;
            list-style: none;
        }

        .node-content {
            display: flex;
            align-items: center;
            padding: 10px;
            margin: 4px 0;
            border-radius: 4px;
            transition: background 0.2s;
            cursor: pointer;
        }

        .node-content:hover {
            background: #f5f5f5;
        }

        .node-content.collapsed {
            opacity: 0.7;
        }

        .toggle {
            width: 20px;
            height: 20px;
            margin-right: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            color: #666;
            cursor: pointer;
        }

        .toggle.empty {
            visibility: hidden;
        }

        .node-name {
            font-weight: 500;
            color: #333;
            margin-right: 12px;
        }

        .node-time {
            font-weight: bold;
            color: #667eea;
            margin-right: 8px;
        }

        .node-percent {
            color: #999;
            font-size: 13px;
            margin-right: 12px;
        }

        .node-bar {
            flex: 1;
            height: 20px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin: 0 12px;
            max-width: 300px;
        }

        .node-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            transition: width 0.3s;
        }

        .node-samples {
            color: #999;
            font-size: 12px;
        }

        .children {
            margin-left: 30px;
            border-left: 2px solid #e0e0e0;
            padding-left: 10px;
        }

        .children.hidden {
            display: none;
        }

        .flamegraph {
            padding: 30px;
            background: white;
        }

        .flame-rect {
            cursor: pointer;
            stroke: white;
            stroke-width: 1;
        }

        .flame-rect:hover {
            stroke: #333;
            stroke-width: 2;
        }

        .flame-text {
            font-size: 11px;
            pointer-events: none;
        }

        .tab-buttons {
            display: flex;
            padding: 0 30px;
            background: white;
            border-bottom: 1px solid #e0e0e0;
        }

        .tab-button {
            padding: 15px 25px;
            background: none;
            border: none;
            border-bottom: 3px solid transparent;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            color: #666;
            transition: all 0.2s;
        }

        .tab-button.active {
            color: #667eea;
            border-bottom-color: #667eea;
        }

        .tab-button:hover {
            color: #333;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        table {
            width: 100%;
            border-collapse: collapse;
        }

        th {
            background: #f8f9fa;
            padding: 12px;
            text-align: left;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #666;
            border-bottom: 2px solid #e0e0e0;
        }

        td {
            padding: 12px;
            border-bottom: 1px solid #f0f0f0;
            font-size: 14px;
        }

        tr:hover {
            background: #f8f9fa;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔥 Profiling Visualizer</h1>
            <div class="meta" id="metadata"></div>
        </div>

        <div class="file-loader" id="fileLoader">
            <h3>📂 Load Profiling Data</h3>

            <div class="file-selector">
                <div class="directory-browser" id="directoryBrowser">
                    <div class="loading-spinner"></div>
                    <div class="current-path" id="currentPath">Loading...</div>
                    <ul class="dir-list" id="dirList"></ul>
                    <ul class="file-list" id="fileList"></ul>
                </div>
                <button class="load-button" id="loadButton" disabled onclick="loadSelectedFile()">Load Selected File</button>
            </div>

            <div class="separator">OR</div>

            <div class="drop-zone" id="dropZone">
                <input type="file" id="fileInput" accept=".json" style="display: none;">
                <div class="drop-zone-text">📁 Drop JSON file here or click to browse</div>
                <div class="drop-zone-hint">Supports local JSON files</div>
            </div>
        </div>

        <div id="contentWrapper" class="content-wrapper">
            <div class="stats" id="stats"></div>

            <div class="tab-buttons">
                <button class="tab-button active" onclick="switchTab('tree')">Tree View</button>
                <button class="tab-button" onclick="switchTab('table')">Table View</button>
                <button class="tab-button" onclick="switchTab('flame')">Flame Graph</button>
            </div>

            <div class="controls">
                <label>
                    Min Time (ms):
                    <input type="number" id="minTime" value="0" step="0.1" onchange="applyFilters()">
                </label>
                <label>
                    <input type="checkbox" id="expandAll" onchange="toggleExpandAll()">
                    Expand All
                </label>
            </div>

            <div id="tree-tab" class="tab-content active">
                <div class="tree-container" id="treeView"></div>
            </div>

            <div id="table-tab" class="tab-content">
                <div class="tree-container" id="tableView"></div>
            </div>

            <div id="flame-tab" class="tab-content">
                <div class="flamegraph" id="flameView"></div>
            </div>
        </div>
    </div>

    <script>
        let profilingData = null;
        let expandedNodes = new Set();
        let minTimeFilter = 0;
        let currentDirectory = '';
        let selectedFilePath = null;

        // Initialize
        window.addEventListener('load', () => {
            setupFileLoader();

            // Auto-load if data is already available
            fetch('/data.json')
                .then(r => {
                    if (r.ok) return r.json();
                    throw new Error('No initial data');
                })
                .then(data => {
                    profilingData = data;
                    showContent();
                    renderAll();
                })
                .catch(() => {
                    // No initial data, show file loader
                    console.log('No initial data, waiting for file selection');
                })
                .finally(() => {
                    // Load directory browser regardless
                    loadDirectory(null);
                });
        });

        function setupFileLoader() {
            const dropZone = document.getElementById('dropZone');
            const fileInput = document.getElementById('fileInput');

            // Click to browse
            dropZone.addEventListener('click', () => fileInput.click());

            // File input change
            fileInput.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) loadFromFile(file);
            });

            // Drag and drop
            dropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropZone.classList.add('drag-over');
            });

            dropZone.addEventListener('dragleave', () => {
                dropZone.classList.remove('drag-over');
            });

            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.classList.remove('drag-over');
                const file = e.dataTransfer.files[0];
                if (file && file.name.endsWith('.json')) {
                    loadFromFile(file);
                } else {
                    alert('Please drop a valid JSON file');
                }
            });
        }

        function loadFromFile(file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const data = JSON.parse(e.target.result);
                    profilingData = data;
                    showContent();
                    renderAll();
                } catch (err) {
                    alert('Error parsing JSON: ' + err.message);
                }
            };
            reader.readAsText(file);
        }

        function loadDirectory(path) {
            // Show loading state
            const browser = document.getElementById('directoryBrowser');
            browser.classList.add('loading');

            const pathDisplay = document.getElementById('currentPath');
            const dirList = document.getElementById('dirList');
            const fileList = document.getElementById('fileList');

            // Clear previous content
            dirList.innerHTML = '';
            fileList.innerHTML = '';

            currentDirectory = path;

            const url = (path === null || path === '') ? '/list_dirs' : `/list_dirs?path=${encodeURIComponent(path)}`;
            const fileUrl = (path === null || path === '') ? '/list_files' : `/list_files?path=${encodeURIComponent(path)}`;

            // Load both directories and files in parallel
            Promise.all([
                fetch(url).then(r => r.json()),
                fetch(fileUrl).then(r => r.json())
            ])
            .then(([dirs, files]) => {
                console.log('Loaded directories:', dirs);
                console.log('Loaded files:', files);

                // Update path display
                if (path) {
                    pathDisplay.textContent = path;
                } else {
                    pathDisplay.textContent = '(current directory)';
                }

                // Render directories
                if (dirs.length === 0) {
                    dirList.innerHTML = '<li style="padding: 10px 15px; color: #999; font-size: 13px;">No subdirectories</li>';
                } else {
                    dirs.forEach(dir => {
                        const li = document.createElement('li');
                        li.className = 'dir-item';
                        li.innerHTML = `
                            <span class="dir-icon">📁</span>
                            <div class="file-info">
                                <div class="dir-name">${dir.name}</div>
                            </div>
                        `;
                        li.onclick = () => loadDirectory(dir.path);
                        dirList.appendChild(li);
                    });
                }

                // Render files
                if (files.length === 0) {
                    fileList.innerHTML = '<li style="padding: 10px 15px; color: #999; font-size: 13px;">No JSON files in this directory</li>';
                } else {
                    files.forEach(file => {
                        const li = document.createElement('li');
                        li.className = 'file-item';
                        const date = new Date(file.mtime * 1000).toLocaleString();
                        const size = (file.size / 1024).toFixed(1);

                        li.innerHTML = `
                            <span class="file-icon">📄</span>
                            <div class="file-info">
                                <div class="file-name">${file.name}</div>
                                <div class="file-meta">${size} KB • ${date}</div>
                            </div>
                        `;
                        li.onclick = () => selectFile(file.path, li);
                        fileList.appendChild(li);
                    });
                }
            })
            .catch(err => {
                console.error('Error loading directory:', err);
                pathDisplay.textContent = 'Error loading directory';
                dirList.innerHTML = '<li style="padding: 15px; color: #f44; font-size: 13px;">Error loading directories</li>';
                fileList.innerHTML = '<li style="padding: 15px; color: #f44; font-size: 13px;">Error loading files</li>';
            })
            .finally(() => {
                // Hide loading state
                browser.classList.remove('loading');
            });
        }

        function selectFile(path, element) {
            // Remove previous selection
            document.querySelectorAll('.file-item').forEach(item => {
                item.classList.remove('selected');
            });

            // Mark as selected
            element.classList.add('selected');
            selectedFilePath = path;
            document.getElementById('loadButton').disabled = false;
        }

        function loadSelectedFile() {
            if (!selectedFilePath) {
                alert('Please select a file');
                return;
            }

            fetch(`/load_file?path=${encodeURIComponent(selectedFilePath)}`)
                .then(r => r.json())
                .then(data => {
                    profilingData = data;
                    showContent();
                    renderAll();
                })
                .catch(err => alert('Error loading file: ' + err.message));
        }

        function showContent() {
            const fileLoader = document.getElementById('fileLoader');
            fileLoader.classList.add('compact');
            fileLoader.querySelector('h3').textContent = '📂 Load Another File';
            document.getElementById('contentWrapper').classList.add('visible');
        }

        function switchTab(tab) {
            document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            event.target.classList.add('active');
            document.getElementById(tab + '-tab').classList.add('active');

            // Re-render flame graph when switching to it (to get correct width)
            if (tab === 'flame') {
                setTimeout(() => renderFlameGraph(), 10);
            }
        }

        function renderAll() {
            renderMetadata();
            renderStats();
            renderTree();
            renderTable();
            renderFlameGraph();
        }

        function renderMetadata() {
            const meta = document.getElementById('metadata');
            meta.innerHTML = `
                <strong>Timestamp:</strong> ${new Date(profilingData.timestamp).toLocaleString()}
            `;
        }

        function renderStats() {
            const stats = document.getElementById('stats');
            const tree = profilingData.profiling_tree;

            let totalTime = 0;
            let totalOps = 0;
            let maxTime = 0;
            let maxOp = '';

            function traverse(nodes) {
                nodes.forEach(node => {
                    totalOps++;
                    totalTime += node.avg_latency_ms;
                    if (node.avg_latency_ms > maxTime) {
                        maxTime = node.avg_latency_ms;
                        maxOp = node.name;
                    }
                    if (node.children) traverse(node.children);
                });
            }

            traverse(tree);

            stats.innerHTML = `
                <div class="stat-card">
                    <div class="label">Total Time</div>
                    <div class="value">${tree[0].avg_latency_ms.toFixed(2)}<span class="unit">ms</span></div>
                </div>
                <div class="stat-card">
                    <div class="label">Operations</div>
                    <div class="value">${totalOps}</div>
                </div>
                <div class="stat-card">
                    <div class="label">Slowest Operation</div>
                    <div class="value">${maxOp}</div>
                    <div style="font-size: 12px; color: #999; margin-top: 4px;">${maxTime.toFixed(2)} ms</div>
                </div>
            `;
        }

        function renderTree() {
            const container = document.getElementById('treeView');
            container.innerHTML = '<ul class="tree-node">' +
                profilingData.profiling_tree.map(node => renderNode(node)).join('') +
                '</ul>';
        }

        function renderNode(node) {
            if (node.avg_latency_ms < minTimeFilter) return '';

            const hasChildren = node.children && node.children.length > 0;
            const nodeId = node.full_path;
            const isExpanded = expandedNodes.has(nodeId);

            // Use percent_of_total if available, otherwise fall back to 100 for root
            const percent = node.percent_of_total !== undefined ? node.percent_of_total.toFixed(1) : '100.0';

            return `
                <li class="tree-node">
                    <div class="node-content">
                        <span class="toggle ${hasChildren ? '' : 'empty'}" onclick="toggleNode('${nodeId}')">
                            ${hasChildren ? (isExpanded ? '▼' : '▶') : ''}
                        </span>
                        <span class="node-name">${node.name}</span>
                        <span class="node-time">${node.avg_latency_ms.toFixed(2)} ms</span>
                        <span class="node-percent">${percent}%</span>
                        <div class="node-bar">
                            <div class="node-bar-fill" style="width: ${Math.min(parseFloat(percent), 100)}%"></div>
                        </div>
                        <span class="node-samples">${node.samples} sample${node.samples > 1 ? 's' : ''}</span>
                    </div>
                    ${hasChildren ? `
                        <ul class="children ${isExpanded ? '' : 'hidden'}" id="children-${nodeId}">
                            ${node.children.map(child => renderNode(child)).join('')}
                        </ul>
                    ` : ''}
                </li>
            `;
        }

        function toggleNode(nodeId) {
            if (expandedNodes.has(nodeId)) {
                expandedNodes.delete(nodeId);
            } else {
                expandedNodes.add(nodeId);
            }
            renderTree();
        }

        function toggleExpandAll() {
            const expandAll = document.getElementById('expandAll').checked;

            function collectAllNodes(nodes) {
                let ids = [];
                nodes.forEach(node => {
                    ids.push(node.full_path);
                    if (node.children) {
                        ids = ids.concat(collectAllNodes(node.children));
                    }
                });
                return ids;
            }

            if (expandAll) {
                expandedNodes = new Set(collectAllNodes(profilingData.profiling_tree));
            } else {
                expandedNodes.clear();
            }
            renderTree();
        }

        function applyFilters() {
            minTimeFilter = parseFloat(document.getElementById('minTime').value) || 0;
            renderTree();
            renderTable();
        }

        function renderTable() {
            const container = document.getElementById('tableView');

            let rows = [];
            function traverse(nodes, depth = 0) {
                nodes.forEach(node => {
                    if (node.avg_latency_ms >= minTimeFilter) {
                        rows.push({...node, depth});
                        if (node.children) traverse(node.children, depth + 1);
                    }
                });
            }
            traverse(profilingData.profiling_tree);

            // Sort by time descending
            rows.sort((a, b) => b.avg_latency_ms - a.avg_latency_ms);

            container.innerHTML = `
                <table>
                    <thead>
                        <tr>
                            <th>Operation</th>
                            <th>Avg Time (ms)</th>
                            <th>Min (ms)</th>
                            <th>Max (ms)</th>
                            <th>Std Dev (ms)</th>
                            <th>Samples</th>
                            <th>% of Total</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${rows.map(row => `
                            <tr>
                                <td style="padding-left: ${row.depth * 20 + 12}px">${row.name}</td>
                                <td><strong>${row.avg_latency_ms.toFixed(3)}</strong></td>
                                <td>${row.min_ms.toFixed(3)}</td>
                                <td>${row.max_ms.toFixed(3)}</td>
                                <td>${row.std_dev_ms.toFixed(3)}</td>
                                <td>${row.samples}</td>
                                <td>${row.percent_of_total ? row.percent_of_total.toFixed(1) + '%' : '-'}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            `;
        }

        function renderFlameGraph() {
            const container = document.getElementById('flameView');

            if (!profilingData || !profilingData.profiling_tree) {
                console.error('No profiling data available for flame graph');
                container.innerHTML = '<div style="padding: 20px; color: #999;">No profiling data loaded</div>';
                return;
            }

            // Get container width, use a minimum if container is hidden
            const containerWidth = container.clientWidth || container.parentElement.clientWidth || 1200;
            const width = Math.max(containerWidth - 60, 600);
            const barHeight = 20;
            const barGap = 1;

            console.log('Rendering flame graph, container width:', containerWidth, 'svg width:', width);
            console.log('Profiling tree:', profilingData.profiling_tree);

            // Calculate total height needed
            function getMaxDepth(node, depth = 0) {
                if (!node.children || node.children.length === 0) return depth;
                return Math.max(...node.children.map(c => getMaxDepth(c, depth + 1)));
            }

            const maxDepth = Math.max(...profilingData.profiling_tree.map(n => getMaxDepth(n))) + 1;
            const height = maxDepth * (barHeight + barGap) + 40;

            console.log('Max depth:', maxDepth, 'Height:', height);

            const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            svg.setAttribute('width', width);
            svg.setAttribute('height', height);
            svg.style.background = '#fff';

            // Calculate total time for root width calculation
            const totalTime = profilingData.profiling_tree.reduce((sum, n) => sum + n.avg_latency_ms, 0);

            function renderFlameNode(node, x, nodeWidth, depth) {
                if (nodeWidth < 0.5) return;

                const y = depth * (barHeight + barGap);

                // Material Design color palette
                const materialColors = [
                    '#EF5350', // Red 400
                    '#EC407A', // Pink 400
                    '#AB47BC', // Purple 400
                    '#7E57C2', // Deep Purple 400
                    '#5C6BC0', // Indigo 400
                    '#42A5F5', // Blue 400
                    '#29B6F6', // Light Blue 400
                    '#26C6DA', // Cyan 400
                    '#26A69A', // Teal 400
                    '#66BB6A', // Green 400
                    '#9CCC65', // Light Green 400
                    '#D4E157', // Lime 400
                    '#FFEE58', // Yellow 400
                    '#FFCA28', // Amber 400
                    '#FFA726', // Orange 400
                    '#FF7043', // Deep Orange 400
                ];

                // Create rect
                const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                rect.setAttribute('x', x);
                rect.setAttribute('y', y);
                rect.setAttribute('width', nodeWidth);
                rect.setAttribute('height', barHeight);
                rect.setAttribute('class', 'flame-rect');

                // Pick color based on depth, cycling through material palette
                const color = materialColors[depth % materialColors.length];
                rect.setAttribute('fill', color);

                // Add tooltip
                const percent = node.percent_of_total ? node.percent_of_total.toFixed(1) : '?';
                const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
                title.textContent = `${node.name}\n${node.avg_latency_ms.toFixed(2)} ms (${percent}%)\n${node.samples} sample${node.samples > 1 ? 's' : ''}`;
                rect.appendChild(title);

                svg.appendChild(rect);

                // Add text if wide enough
                if (nodeWidth > 40) {
                    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    text.setAttribute('x', x + 4);
                    text.setAttribute('y', y + barHeight / 2 + 4);
                    text.setAttribute('class', 'flame-text');
                    text.setAttribute('fill', '#fff');
                    text.setAttribute('font-size', '11px');
                    text.setAttribute('font-family', 'sans-serif');

                    // Truncate text to fit
                    const maxChars = Math.floor(nodeWidth / 7);
                    const displayText = node.name.length > maxChars
                        ? node.name.substring(0, maxChars - 1) + '…'
                        : node.name;
                    text.textContent = displayText;
                    svg.appendChild(text);
                }

                // Render children
                if (node.children && node.children.length > 0) {
                    let childX = x;
                    const nodeTime = node.avg_latency_ms;

                    node.children.forEach(child => {
                        // Calculate child width proportional to parent width
                        const childProportion = child.avg_latency_ms / nodeTime;
                        const childWidth = nodeWidth * childProportion;

                        renderFlameNode(child, childX, childWidth, depth + 1);
                        childX += childWidth;
                    });
                }
            }

            // Render all root nodes
            let currentX = 0;
            profilingData.profiling_tree.forEach(node => {
                const nodeWidth = (node.avg_latency_ms / totalTime) * width;
                console.log(`Rendering root node: ${node.name}, width: ${nodeWidth}, x: ${currentX}`);
                renderFlameNode(node, currentX, nodeWidth, 0);
                currentX += nodeWidth;
            });

            console.log('Total SVG children:', svg.childNodes.length);

            container.innerHTML = '';
            container.appendChild(svg);

            console.log('Flame graph rendered successfully');
        }
    </script>
</body>
</html>
"""


def main():
    """Main entry point."""
    # Optional: Load initial file if provided
    if len(sys.argv) >= 2:
        json_path = Path(sys.argv[1])
        if not json_path.exists():
            print(f"Error: File not found: {json_path}")
            sys.exit(1)

        # Load profiling data
        with open(json_path, "r", encoding="utf-8") as f:
            ProfileVisualizerHandler.profiling_data = json.load(f)

        # Set search directory to the parent directory of the provided file
        ProfileVisualizerHandler.search_directory = str(json_path.parent.absolute())
        initial_file = json_path.name
    else:
        # No file provided, use parent directory of script location
        script_path = Path(__file__).resolve()
        ProfileVisualizerHandler.search_directory = str(script_path.parent.parent)
        initial_file = None

    # Set root directory for relative path calculations
    ProfileVisualizerHandler.root_directory = ProfileVisualizerHandler.search_directory

    # Start server
    port = 8000
    server = HTTPServer(("localhost", port), ProfileVisualizerHandler)

    print(f"\n{'='*60}")
    print(f"🔥 Profiling Visualizer Started!")
    print(f"{'='*60}")
    print(f"\nCurrent working directory: {os.getcwd()}")
    if initial_file:
        print(f"\nInitially loaded: {initial_file}")
        if ProfileVisualizerHandler.profiling_data:
            print(f"Timestamp: {ProfileVisualizerHandler.profiling_data.get('timestamp', 'N/A')}")
    print(f"\nSearch directory: {ProfileVisualizerHandler.search_directory}")
    print(f"\n👉 Open in browser: http://localhost:{port}")
    print(f"\nYou can:")
    print(f"  • Select files from the dropdown (scans: {ProfileVisualizerHandler.search_directory})")
    print(f"  • Drag & drop JSON files into the browser")
    print(f"  • Click the drop zone to browse files")
    print(f"\nPress Ctrl+C to stop the server")
    print(f"{'='*60}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped")


if __name__ == "__main__":
    main()
