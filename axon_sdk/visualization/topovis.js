fetch('/graph_data')
    .then(response => response.json())
    .then(data => {
        var g = new dagre.graphlib.Graph({ compound: true,  multigraph: true });
        g.setGraph({ rankdir: "LR" });
        g.setDefaultEdgeLabel(function() { return {}; });

        data.nodes.forEach(node => {
            g.setNode(node.id, { label: node.label, width: 250, height: 250 });
        });

        data.edges.forEach(edge => {
            g.setEdge(edge.source, edge.target, { label: edge.label, color: edge.color, curve: d3.curveBasis }, edge.uid);
        });

        if (data.groups && data.groups.length > 0) {
            data.groups.forEach(group => {
                // group label is added later on for precise positioning
                g.setNode(group.id, { label: "", width: 600, height: 300, style: "fill:rgb(240, 240, 240); stroke: #333; stroke-width: 2;" });
                group.nodes.forEach(nodeId => {
                    g.setParent(nodeId, group.id);
                });
            });
        }

        var render = new dagreD3.render();
        var svg = d3.select("#graph").append("svg")
            .attr("width", "100%")
            .attr("height", "100%");

        var inner = svg.append("g");
        var zoom = d3.zoom()
            .scaleExtent([0.1, 3])
            .on("zoom", function(event) {
                inner.attr("transform", event.transform);
            });

        svg.call(zoom);
        render(inner, g);

        var xCenterOffset = (svg.attr("width") - g.graph().width) / 2;
        var yCenterOffset = (svg.attr("height") - g.graph().height) / 2;
        inner.attr("transform", "translate(" + xCenterOffset + ", " + yCenterOffset + ")");

        // Style nodes
        inner.selectAll("g.node rect")
            .style("fill", "#FFFFFF80")
            .style("stroke", "#333")
            .style("stroke-width", 2);

        // Style edges
        inner.selectAll("g.edgePath path")
            .style("stroke", d => data.edges.find(edge => edge.uid === d.name).color)
            // .style("stroke", "#f0f0f0")
            .style("stroke-width", 2)
            .style("stroke-opacity", 0.5)
            .style("fill", "none");

        // Style labels to edges
        inner.selectAll("g.edgeLabel text")
            .style("fill", "#333")
            .style("font-size", "12px")
            .text(d => data.edges.find(edge => edge.uid === d.name).label);

        // Style for group label
        inner.selectAll("g.cluster").each(function(d, i) {
            const bbox = this.getBBox();  // Need the renderedd box to know actual width and height

            const this_group = data.groups.find(group => group.id === this.__data__)
            const this_group_node = g.node(this_group.id)
            
            const labelX = this_group_node.x;  // for some reason, x and y must be retrieved from the group node
            const labelY = this_group_node.y - bbox.height / 2 - 10;
            
            // Center text above group
            inner.append("text")
                .attr("x", labelX)
                .attr("y", labelY)
                .attr("text-anchor", "middle")
                .style("font-size", "20px")
                .style("font-weight", "bold")
                .text(d => this_group.label);
            });

    })
    .catch(error => console.error('Error fetching graph data:', error));
