console.log('Node editor script starting execution...');

class NodeEditor {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.canvas = document.getElementById('canvas');
        this.palette = document.getElementById('palette');
        this.inspector = document.getElementById('inspector');
        this.nodes = [];
        this.connections = [];
        this.selectedNode = null;
        this.draggingNode = null;
        this.connecting = null;
        this.nodeIdCounter = 0;
        this.offset = { x: 0, y: 0 };

        this.init();
    }

    init() {
        // Set up SVG for connections
        this.svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        this.svg.style.position = 'absolute';
        this.svg.style.width = '100%';
        this.svg.style.height = '100%';
        this.svg.style.pointerEvents = 'none';
        this.svg.style.zIndex = '1';
        this.svg.style.top = '0';
        this.svg.style.left = '0';
        this.canvas.appendChild(this.svg);

        // Set up event listeners
        this.setupPalette();
        this.setupCanvas();

        // Add gradient input node by default
        this.addNode('gradient_input', 100, 300);
    }

    setupPalette() {
        // Make palette items draggable
        const items = this.palette.querySelectorAll('.palette-item');
        items.forEach(item => {
            item.draggable = true;
            item.addEventListener('dragstart', (e) => this.onPaletteDragStart(e));
            item.addEventListener('dragend', (e) => this.onPaletteDragEnd(e));
        });
    }

    setupCanvas() {
        this.canvas.addEventListener('dragover', (e) => e.preventDefault());
        this.canvas.addEventListener('drop', (e) => this.onCanvasDrop(e));
        this.canvas.addEventListener('mousedown', (e) => this.onCanvasMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.onCanvasMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.onCanvasMouseUp(e));
    }

    onPaletteDragStart(e) {
        e.target.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'copy';
        e.dataTransfer.setData('nodeType', e.target.dataset.type);
    }

    onPaletteDragEnd(e) {
        e.target.classList.remove('dragging');
    }

    onCanvasDrop(e) {
        e.preventDefault();
        const nodeType = e.dataTransfer.getData('nodeType');
        if (nodeType) {
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left - 80;
            const y = e.clientY - rect.top - 30;
            this.addNode(nodeType, x, y);
            this.updateCodePreview();
        }
    }

    addNode(type, x, y) {
        console.log(`Adding node: type=${type}, x=${x}, y=${y}`);
        const nodeId = `node-${this.nodeIdCounter++}`;
        const nodeData = window.CHAINABLE_FUNCTIONS[type] || {
            description: type === 'gradient_input' ? 'Gradient Input' : 'Unknown',
            color: '#666',
            inputs: type === 'gradient_input' ? [] : ['grad'],
            outputs: type === 'gradient_input' ? ['grad'] : []
        };

        const node = document.createElement('div');
        node.className = 'node';
        node.id = nodeId;
        node.style.left = x + 'px';
        node.style.top = y + 'px';
        node.style.setProperty('--node-color', nodeData.color);
        node.style.borderLeftColor = nodeData.color;
        node.style.borderLeftWidth = '4px';
        console.log('Created node element:', node);

        // Special styling for gradient input
        if (type === 'gradient_input') {
            node.classList.add('gradient-input');
        }

        node.innerHTML = `
            <div class="node-title">${type === 'gradient_input' ? 'Gradient Input' : type.replace(/_/g, ' ')}</div>
            <div class="node-description">${nodeData.description}</div>
            ${nodeData.inputs.length ? '<div class="node-port node-port-input"></div>' : ''}
            ${nodeData.outputs.length ? '<div class="node-port node-port-output"></div>' : ''}
        `;

        console.log('Canvas element:', this.canvas);
        console.log('Canvas dimensions:', this.canvas.offsetWidth, 'x', this.canvas.offsetHeight);
        this.canvas.appendChild(node);
        console.log('Node appended to canvas. Total nodes in canvas:', this.canvas.querySelectorAll('.node').length);

        // Set up node interactions
        node.addEventListener('mousedown', (e) => this.onNodeMouseDown(e, nodeId));

        // Set up port interactions
        const inputPort = node.querySelector('.node-port-input');
        const outputPort = node.querySelector('.node-port-output');

        if (inputPort) {
            inputPort.addEventListener('mousedown', (e) => this.onPortMouseDown(e, nodeId, 'input'));
        }
        if (outputPort) {
            outputPort.addEventListener('mousedown', (e) => this.onPortMouseDown(e, nodeId, 'output'));
        }

        // Store node data
        this.nodes.push({
            id: nodeId,
            type: type,
            element: node,
            x: x,
            y: y,
            params: nodeData.params || {}
        });

        return nodeId;
    }

    onNodeMouseDown(e, nodeId) {
        if (e.target.classList.contains('node-port')) return;

        e.preventDefault();
        this.selectNode(nodeId);

        const node = this.nodes.find(n => n.id === nodeId);
        this.draggingNode = node;

        const rect = node.element.getBoundingClientRect();
        const canvasRect = this.canvas.getBoundingClientRect();
        this.offset = {
            x: e.clientX - rect.left + canvasRect.left,
            y: e.clientY - rect.top + canvasRect.top
        };
    }

    onPortMouseDown(e, nodeId, portType) {
        e.preventDefault();
        e.stopPropagation();

        const port = e.target;
        const rect = port.getBoundingClientRect();
        const canvasRect = this.canvas.getBoundingClientRect();

        this.connecting = {
            nodeId: nodeId,
            portType: portType,
            startX: rect.left + rect.width / 2 - canvasRect.left,
            startY: rect.top + rect.height / 2 - canvasRect.top
        };

        // Create preview connection
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        line.classList.add('connection-preview');
        line.id = 'preview-connection';
        this.svg.appendChild(line);
    }

    onCanvasMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        if (this.draggingNode) {
            this.draggingNode.element.style.left = (x - this.offset.x) + 'px';
            this.draggingNode.element.style.top = (y - this.offset.y) + 'px';
            this.draggingNode.x = x - this.offset.x;
            this.draggingNode.y = y - this.offset.y;
            this.updateConnections();
        }

        if (this.connecting) {
            const preview = document.getElementById('preview-connection');
            if (preview) {
                const path = this.createConnectionPath(
                    this.connecting.startX,
                    this.connecting.startY,
                    x,
                    y
                );
                preview.setAttribute('d', path);
            }
        }
    }

    onCanvasMouseUp(e) {
        if (this.connecting) {
            const preview = document.getElementById('preview-connection');
            if (preview) preview.remove();

            // Check if we're over a port
            const target = document.elementFromPoint(e.clientX, e.clientY);
            if (target && target.classList.contains('node-port')) {
                const targetNode = target.closest('.node');
                const targetNodeId = targetNode.id;
                const targetPortType = target.classList.contains('node-port-input') ? 'input' : 'output';

                // Validate connection
                if (this.canConnect(this.connecting.nodeId, this.connecting.portType, targetNodeId, targetPortType)) {
                    this.addConnection(this.connecting.nodeId, this.connecting.portType, targetNodeId, targetPortType);
                    this.updateCodePreview();
                }
            }

            this.connecting = null;
        }

        this.draggingNode = null;
    }

    canConnect(fromNodeId, fromPortType, toNodeId, toPortType) {
        // Can't connect to same node
        if (fromNodeId === toNodeId) return false;

        // Must connect output to input
        if (fromPortType === toPortType) return false;

        // Check if connection already exists
        const exists = this.connections.some(c =>
            (c.from.nodeId === fromNodeId && c.to.nodeId === toNodeId) ||
            (c.from.nodeId === toNodeId && c.to.nodeId === fromNodeId)
        );

        return !exists;
    }

    addConnection(fromNodeId, fromPortType, toNodeId, toPortType) {
        // Ensure output connects to input
        if (fromPortType === 'input') {
            [fromNodeId, toNodeId] = [toNodeId, fromNodeId];
            [fromPortType, toPortType] = [toPortType, fromPortType];
        }

        const connection = {
            from: { nodeId: fromNodeId, portType: fromPortType },
            to: { nodeId: toNodeId, portType: toPortType },
            element: null
        };

        const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        line.classList.add('connection');
        line.classList.add('animated');
        this.svg.appendChild(line);

        connection.element = line;
        this.connections.push(connection);

        this.updateConnections();
    }

    updateConnections() {
        this.connections.forEach(conn => {
            const fromNode = this.nodes.find(n => n.id === conn.from.nodeId);
            const toNode = this.nodes.find(n => n.id === conn.to.nodeId);

            if (fromNode && toNode) {
                const fromPort = fromNode.element.querySelector('.node-port-output');
                const toPort = toNode.element.querySelector('.node-port-input');

                if (fromPort && toPort) {
                    const fromRect = fromPort.getBoundingClientRect();
                    const toRect = toPort.getBoundingClientRect();
                    const canvasRect = this.canvas.getBoundingClientRect();

                    const x1 = fromRect.left + fromRect.width / 2 - canvasRect.left;
                    const y1 = fromRect.top + fromRect.height / 2 - canvasRect.top;
                    const x2 = toRect.left + toRect.width / 2 - canvasRect.left;
                    const y2 = toRect.top + toRect.height / 2 - canvasRect.top;

                    const path = this.createConnectionPath(x1, y1, x2, y2);
                    conn.element.setAttribute('d', path);

                    // Store path coordinates for particle animation
                    conn.pathCoords = { x1, y1, x2, y2 };
                }
            }
        });
    }

    // Animate gradient flow particles
    startGradientFlow() {
        if (this.flowInterval) return;

        this.flowInterval = setInterval(() => {
            this.connections.forEach(conn => {
                if (conn.pathCoords) {
                    this.createFlowParticle(conn.pathCoords);
                }
            });
        }, 500);
    }

    stopGradientFlow() {
        if (this.flowInterval) {
            clearInterval(this.flowInterval);
            this.flowInterval = null;
        }

        // Remove all particles
        const particles = this.canvas.querySelectorAll('.flow-particle');
        particles.forEach(p => p.remove());
    }

    createFlowParticle(coords) {
        const particle = document.createElement('div');
        particle.className = 'flow-particle';
        particle.style.left = coords.x1 + 'px';
        particle.style.top = coords.y1 + 'px';
        this.canvas.appendChild(particle);

        // Animate along bezier curve
        const duration = 2000;
        const startTime = Date.now();

        const animate = () => {
            const elapsed = Date.now() - startTime;
            const t = Math.min(elapsed / duration, 1);

            if (t >= 1) {
                particle.remove();
                return;
            }

            // Calculate position on bezier curve
            const dx = Math.abs(coords.x2 - coords.x1);
            const cp1x = coords.x1 + dx * 0.5;
            const cp2x = coords.x2 - dx * 0.5;

            // Bezier curve formula
            const x = Math.pow(1-t, 3) * coords.x1 +
                     3 * Math.pow(1-t, 2) * t * cp1x +
                     3 * (1-t) * Math.pow(t, 2) * cp2x +
                     Math.pow(t, 3) * coords.x2;

            const y = Math.pow(1-t, 3) * coords.y1 +
                     3 * Math.pow(1-t, 2) * t * coords.y1 +
                     3 * (1-t) * Math.pow(t, 2) * coords.y2 +
                     Math.pow(t, 3) * coords.y2;

            particle.style.left = x + 'px';
            particle.style.top = y + 'px';
            particle.style.opacity = Math.sin(t * Math.PI);

            requestAnimationFrame(animate);
        };

        requestAnimationFrame(animate);
    }

    createConnectionPath(x1, y1, x2, y2) {
        const dx = Math.abs(x2 - x1);
        const cp1x = x1 + dx * 0.5;
        const cp2x = x2 - dx * 0.5;
        return `M ${x1} ${y1} C ${cp1x} ${y1}, ${cp2x} ${y2}, ${x2} ${y2}`;
    }

    selectNode(nodeId) {
        // Deselect previous
        if (this.selectedNode) {
            this.selectedNode.element.classList.remove('selected');
        }

        // Select new
        const node = this.nodes.find(n => n.id === nodeId);
        if (node) {
            node.element.classList.add('selected');
            this.selectedNode = node;
            this.showInspector(node);
        }
    }

    showInspector(node) {
        this.inspector.classList.add('active');

        const nodeData = window.CHAINABLE_FUNCTIONS[node.type];
        if (!nodeData || !nodeData.params) return;

        // Build inspector content
        let html = `<h3>${node.type.replace(/_/g, ' ')}</h3>`;

        Object.entries(nodeData.params).forEach(([key, defaultValue]) => {
            const value = node.params[key] || defaultValue;
            html += `
                <div class="inspector-field">
                    <div class="inspector-label">${key}</div>
                    <input class="inspector-input"
                           type="${typeof defaultValue === 'number' ? 'number' : 'text'}"
                           data-param="${key}"
                           value="${value}"
                           step="${typeof defaultValue === 'number' ? '0.001' : ''}"
                    />
                </div>
            `;
        });

        this.inspector.innerHTML = html;

        // Set up parameter change listeners
        this.inspector.querySelectorAll('.inspector-input').forEach(input => {
            input.addEventListener('change', (e) => {
                const param = e.target.dataset.param;
                let value = e.target.value;

                // Convert to appropriate type
                if (e.target.type === 'number') {
                    value = parseFloat(value);
                }

                node.params[param] = value;
                this.updateCodePreview();
            });
        });
    }

    updateCodePreview() {
        const codePreview = document.getElementById('code-output');
        if (!codePreview) return;

        // Generate Python code from pipeline
        const pipelineData = this.exportPipeline();
        const nodes = pipelineData.nodes.filter(n => n.type !== 'gradient_input');

        if (nodes.length === 0) {
            codePreview.textContent = '# Add optimizer components to build your pipeline';
            return;
        }

        let code = 'optimizer = BaseOpt(\n';
        code += '    model.parameters(),\n';
        code += '    lr=0.001,\n';

        // Add relevant parameters
        const params = {};
        nodes.forEach(node => {
            Object.assign(params, node.params);
        });

        Object.entries(params).forEach(([key, value]) => {
            if (typeof value === 'string') {
                code += `    ${key}="${value}",\n`;
            } else {
                code += `    ${key}=${value},\n`;
            }
        });

        code += '    fns=[\n';
        nodes.forEach(node => {
            code += `        ${node.type},\n`;
        });
        code += '    ]\n';
        code += ')';

        codePreview.textContent = code;

        // Update hidden pipeline data
        const pipelineInput = document.getElementById('pipeline-data');
        if (pipelineInput) {
            pipelineInput.value = JSON.stringify(pipelineData);
        }
    }

    exportPipeline() {
        return {
            nodes: this.nodes.map(n => ({
                id: n.id,
                type: n.type,
                x: n.x,
                y: n.y,
                params: n.params
            })),
            connections: this.connections.map(c => ({
                from: c.from,
                to: c.to
            }))
        };
    }

    loadRecipe(recipe) {
        // Clear existing nodes except gradient input
        this.nodes = this.nodes.filter(n => n.type === 'gradient_input');
        this.connections = [];
        this.selectedNode = null;

        // Clear canvas except gradient input
        const nodesToRemove = this.canvas.querySelectorAll('.node:not([id="node-0"])');
        nodesToRemove.forEach(n => n.remove());

        // Clear connections
        this.svg.innerHTML = '';

        // Add nodes from recipe
        let lastNodeId = 'node-0'; // gradient input
        let x = 300;
        const y = 300;

        recipe.forEach((item, index) => {
            const nodeId = this.addNode(item.name, x, y);
            const node = this.nodes.find(n => n.id === nodeId);

            // Set parameters
            if (item.params) {
                node.params = item.params;
            }

            // Connect to previous node
            this.addConnection(lastNodeId, 'output', nodeId, 'input');

            lastNodeId = nodeId;
            x += 200;
        });

        this.updateCodePreview();
    }
}

// Initialize when DOM is ready
function initializeNodeEditor() {
    try {
        console.log('Attempting to initialize NodeEditor...');
        const editorElement = document.getElementById('node-editor');
        if (!editorElement) {
            console.error('node-editor element not found! Retrying in 100ms...');
            setTimeout(initializeNodeEditor, 100);
            return;
        }
        console.log('Found node-editor element:', editorElement);
        console.log('Editor dimensions:', editorElement.offsetWidth, 'x', editorElement.offsetHeight);

        if (window.nodeEditor) {
            console.log('NodeEditor already initialized');
            return;
        }

        window.nodeEditor = new NodeEditor('node-editor');
        console.log('NodeEditor initialized successfully');
    } catch (error) {
        console.error('Error initializing NodeEditor:', error);
    }
}

// Try multiple initialization methods
document.addEventListener('DOMContentLoaded', initializeNodeEditor);
// Also try after a delay in case content loads after DOMContentLoaded
setTimeout(initializeNodeEditor, 100);
setTimeout(initializeNodeEditor, 500);
setTimeout(initializeNodeEditor, 1000);
