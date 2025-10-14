#!/usr/bin/env python3
"""
Interactive Gantt-chart style visualization for operation execution.

Creates an HTML visualization where:
- X-axis: Time (milliseconds)
- Y-axis: Concurrent execution lanes (minimal height)
- Operations are shown as rectangles
- Operations that don't overlap in time share the same lane
- Hovering shows operation details and tensor information

Usage:
    python visualize_gantt.py <profile.json> [output.html]
"""

import json
import sys
from datetime import datetime
from pathlib import Path


def parse_timestamp(ts_str: str) -> float:
    """Parse ISO timestamp to seconds since epoch."""
    dt = datetime.fromisoformat(ts_str)
    return dt.timestamp()


def assign_lanes(operations):
    """
    Assign vertical lanes to operations to minimize chart height.

    Operations that don't overlap in time can share the same lane.
    """
    lanes = []

    for op in operations:
        # Find a lane where this operation can fit
        placed = False
        for lane in lanes:
            # Check if this operation overlaps with any operation in this lane
            overlaps = False
            for existing_op in lane:
                if not (op['end_time'] <= existing_op['start_time'] or
                       op['start_time'] >= existing_op['end_time']):
                    overlaps = True
                    break

            if not overlaps:
                lane.append(op)
                op['lane'] = len(lanes) - 1
                placed = True
                break

        if not placed:
            # Create a new lane
            lanes.append([op])
            op['lane'] = len(lanes) - 1

    return len(lanes)


def create_gantt_html(profile_path: str, output_path: str = None):
    """Create interactive Gantt-chart HTML visualization."""

    # Load profile
    with open(profile_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    operation_log = data.get('operation_log', [])

    if not operation_log:
        print("Error: No operation log found in profile", file=sys.stderr)
        return

    print(f"Loaded {len(operation_log)} operations")

    # Parse operations and calculate times
    operations = []
    timestamps = []

    for op in operation_log:
        ts = parse_timestamp(op['timestamp'])
        timestamps.append(ts)

    start_time = min(timestamps)

    # Estimate operation duration (time until next operation or minimal duration)
    for i, op in enumerate(operation_log):
        ts = parse_timestamp(op['timestamp'])
        relative_start = (ts - start_time) * 1000  # Convert to ms

        # Estimate duration: time until next operation or 0.1ms minimum
        if i < len(operation_log) - 1:
            next_ts = parse_timestamp(operation_log[i + 1]['timestamp'])
            duration = max((next_ts - ts) * 1000, 0.1)  # At least 0.1ms
        else:
            duration = 1.0  # Last operation gets 1ms

        operations.append({
            'id': op['operation_id'],
            'name': op['name'],
            'start_time': relative_start,
            'end_time': relative_start + duration,
            'duration': duration,
            'input_tensor_ids': op.get('input_tensor_ids', []),
            'output_tensor_ids': op.get('output_tensor_ids', []),
            'input_shapes': op.get('input_shapes', []),
            'output_shape': op.get('output_shape', None),
            'module_type': op.get('module_type', ''),
        })

    # Assign lanes
    num_lanes = assign_lanes(operations)
    total_time = max(op['end_time'] for op in operations)

    print(f"Timeline span: {total_time:.2f} ms")
    print(f"Assigned {num_lanes} concurrent execution lanes")

    # Generate HTML
    html = generate_html(operations, num_lanes, total_time)

    # Save
    if output_path is None:
        output_path = profile_path.replace('.json', '_gantt.html')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nVisualization saved to: {output_path}")
    print(f"Open in browser: file://{Path(output_path).absolute()}")


def generate_html(operations, num_lanes, total_time):
    """Generate the HTML content."""

    # Prepare operation data for JavaScript
    ops_json = json.dumps(operations, indent=2)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Operation Timeline - Gantt Chart</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}

        h1 {{
            color: #333;
            margin-bottom: 10px;
        }}

        .stats {{
            color: #666;
            margin-bottom: 20px;
            font-size: 14px;
        }}

        .chart-container {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            overflow-x: auto;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        svg {{
            display: block;
        }}

        .operation-rect {{
            cursor: pointer;
            transition: opacity 0.2s;
        }}

        .operation-rect:hover {{
            opacity: 0.8;
            stroke: #333;
            stroke-width: 2;
        }}

        .tooltip {{
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 12px;
            border-radius: 6px;
            font-size: 13px;
            pointer-events: none;
            display: none;
            z-index: 1000;
            max-width: 400px;
            line-height: 1.6;
        }}

        .tooltip-title {{
            font-weight: bold;
            margin-bottom: 8px;
            color: #4CAF50;
            font-size: 14px;
        }}

        .tooltip-row {{
            margin: 4px 0;
        }}

        .tooltip-label {{
            color: #aaa;
            display: inline-block;
            width: 120px;
        }}

        .axis {{
            font-size: 12px;
        }}

        .axis-label {{
            font-size: 14px;
            font-weight: bold;
        }}

        .grid-line {{
            stroke: #eee;
            stroke-width: 1;
        }}

        .legend {{
            margin-top: 20px;
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
        }}

        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 3px;
            border: 1px solid #ddd;
        }}

        /* Loading spinner styles */
        .loading-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(255, 255, 255, 0.95);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 9999;
        }}

        .spinner {{
            width: 60px;
            height: 60px;
            border: 6px solid #f3f3f3;
            border-top: 6px solid #4CAF50;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}

        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}

        .loading-text {{
            margin-top: 20px;
            font-size: 16px;
            color: #666;
            font-weight: 500;
        }}

        .hidden {{
            display: none !important;
        }}
    </style>
</head>
<body>
    <!-- Loading spinner -->
    <div class="loading-overlay" id="loading-overlay">
        <div class="spinner"></div>
        <div class="loading-text">Loading visualization...</div>
    </div>

    <h1>Operation Timeline - Gantt Chart</h1>
    <div class="stats">
        <strong>{len(operations)}</strong> operations |
        <strong>{num_lanes}</strong> concurrent lanes |
        <strong>{total_time:.2f}</strong> ms total time
    </div>

    <div class="chart-container">
        <svg id="gantt-chart"></svg>
    </div>

    <div class="legend" id="legend"></div>

    <div class="tooltip" id="tooltip"></div>

    <!-- D3.js library - Load FIRST -->
    <script src="https://d3js.org/d3.v7.min.js"></script>

    <script>
        const operations = {ops_json};
        const numLanes = {num_lanes};
        const totalTime = {total_time};

        // Configuration
        const margin = {{top: 40, right: 40, bottom: 60, left: 80}};
        const laneHeight = 40;
        const chartHeight = numLanes * laneHeight;
        const chartWidth = Math.max(1200, totalTime * 2);  // 2px per ms minimum
        const width = chartWidth + margin.left + margin.right;
        const height = chartHeight + margin.top + margin.bottom;

        // Color palette for different operation types
        const colorMap = {{
            'Linear': '#4CAF50',
            'RMSNorm': '#2196F3',
            'SiLU': '#FF9800',
            'Embedding': '#9C27B0',
            'default': '#757575'
        }};

        // Create SVG
        const svg = d3.select('#gantt-chart')
            .attr('width', width)
            .attr('height', height);

        const g = svg.append('g')
            .attr('transform', `translate(${{margin.left}},${{margin.top}})`);

        // Scales
        const xScale = d3.scaleLinear()
            .domain([0, totalTime])
            .range([0, chartWidth]);

        const yScale = d3.scaleLinear()
            .domain([0, numLanes])
            .range([0, chartHeight]);

        // Grid lines
        const gridLines = g.append('g').attr('class', 'grid');

        // Vertical grid lines every 100ms
        for (let t = 0; t <= totalTime; t += 100) {{
            gridLines.append('line')
                .attr('class', 'grid-line')
                .attr('x1', xScale(t))
                .attr('x2', xScale(t))
                .attr('y1', 0)
                .attr('y2', chartHeight);
        }}

        // Horizontal grid lines
        for (let lane = 0; lane <= numLanes; lane++) {{
            gridLines.append('line')
                .attr('class', 'grid-line')
                .attr('x1', 0)
                .attr('x2', chartWidth)
                .attr('y1', yScale(lane))
                .attr('y2', yScale(lane));
        }}

        // Draw operations
        const opsGroup = g.append('g').attr('class', 'operations');

        operations.forEach(op => {{
            const color = colorMap[op.module_type] || colorMap.default;
            const x = xScale(op.start_time);
            const y = yScale(op.lane) + 5;
            const width = Math.max(xScale(op.duration), 2);
            const height = laneHeight - 10;

            const rect = opsGroup.append('rect')
                .attr('class', 'operation-rect')
                .attr('x', x)
                .attr('y', y)
                .attr('width', width)
                .attr('height', height)
                .attr('fill', color)
                .attr('rx', 3);

            // Add event listeners
            rect.on('mouseover', function(event) {{
                showTooltip(event, op);
            }});

            rect.on('mouseout', function() {{
                hideTooltip();
            }});
        }});

        // X-axis
        const xAxis = d3.axisBottom(xScale)
            .ticks(10)
            .tickFormat(d => d + ' ms');

        g.append('g')
            .attr('class', 'axis')
            .attr('transform', `translate(0,${{chartHeight}})`)
            .call(xAxis);

        // X-axis label
        g.append('text')
            .attr('class', 'axis-label')
            .attr('x', chartWidth / 2)
            .attr('y', chartHeight + 50)
            .attr('text-anchor', 'middle')
            .text('Time (milliseconds)');

        // Y-axis
        const yAxis = d3.axisLeft(yScale)
            .ticks(numLanes)
            .tickFormat(d => 'Lane ' + Math.floor(d));

        g.append('g')
            .attr('class', 'axis')
            .call(yAxis);

        // Y-axis label
        g.append('text')
            .attr('class', 'axis-label')
            .attr('transform', 'rotate(-90)')
            .attr('x', -chartHeight / 2)
            .attr('y', -60)
            .attr('text-anchor', 'middle')
            .text('Execution Lane');

        // Tooltip functions
        function showTooltip(event, op) {{
            const tooltip = document.getElementById('tooltip');

            const inputShapes = op.input_shapes.map(s => '[' + s.join('x') + ']').join(', ');
            const outputShape = op.output_shape ? '[' + op.output_shape.join('x') + ']' : 'N/A';

            tooltip.innerHTML = `
                <div class="tooltip-title">${{op.name}}</div>
                <div class="tooltip-row">
                    <span class="tooltip-label">Operation ID:</span> ${{op.id}}
                </div>
                <div class="tooltip-row">
                    <span class="tooltip-label">Type:</span> ${{op.module_type}}
                </div>
                <div class="tooltip-row">
                    <span class="tooltip-label">Start Time:</span> ${{op.start_time.toFixed(2)}} ms
                </div>
                <div class="tooltip-row">
                    <span class="tooltip-label">Duration:</span> ${{op.duration.toFixed(2)}} ms
                </div>
                <div class="tooltip-row">
                    <span class="tooltip-label">Lane:</span> ${{op.lane}}
                </div>
                <div class="tooltip-row">
                    <span class="tooltip-label">Input Tensors:</span> ${{op.input_tensor_ids.length}}
                </div>
                <div class="tooltip-row">
                    <span class="tooltip-label">Input Shapes:</span> ${{inputShapes || 'N/A'}}
                </div>
                <div class="tooltip-row">
                    <span class="tooltip-label">Output Tensors:</span> ${{op.output_tensor_ids.length}}
                </div>
                <div class="tooltip-row">
                    <span class="tooltip-label">Output Shape:</span> ${{outputShape}}
                </div>
            `;

            tooltip.style.display = 'block';
            tooltip.style.left = (event.pageX + 15) + 'px';
            tooltip.style.top = (event.pageY + 15) + 'px';
        }}

        function hideTooltip() {{
            document.getElementById('tooltip').style.display = 'none';
        }}

        // Create legend
        const legend = document.getElementById('legend');
        Object.entries(colorMap).forEach(([type, color]) => {{
            if (type !== 'default') {{
                const item = document.createElement('div');
                item.className = 'legend-item';
                item.innerHTML = `
                    <div class="legend-color" style="background-color: ${{color}}"></div>
                    <span>${{type}}</span>
                `;
                legend.appendChild(item);
            }}
        }});

        // Hide loading spinner after chart is rendered
        document.getElementById('loading-overlay').classList.add('hidden');
    </script>
</body>
</html>
"""

    return html


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Error: Missing JSON file argument", file=sys.stderr)
        print(f"\nUsage: {sys.argv[0]} <profile.json> [output.html]", file=sys.stderr)
        print(f"\nExample: {sys.argv[0]} memory_profile.json gantt.html", file=sys.stderr)
        sys.exit(1)

    profile_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(profile_path).exists():
        print(f"Error: File '{profile_path}' not found", file=sys.stderr)
        sys.exit(1)

    create_gantt_html(profile_path, output_path)


if __name__ == '__main__':
    main()
