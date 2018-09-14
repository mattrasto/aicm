$(document).ready(function() {

    $("#sidebar").resizable({
            handles: "w",
            minWidth: 200,
            maxWidth: 1000,
            resize: function(event, ui) {
                $("#viz-container").width($(window).width() - $("#sidebar").width() - 102);
                // Prevents sidebar from sliding left / right
                $("#sidebar").css("right", 0);
                $("#sidebar").css("left", 0);
                // console.log("window: " + $(window).width());
                // console.log("sidebar: " + $("#sidebar").width());
                // console.log("window - sidebar: " + ($(window).width() - $("#sidebar").width()));
                // console.log("viz-container: " + $("#viz-container").width());
            },
            stop: function(event, ui) {}
    });

    settings = {
        'center_root': true,
        'link_strength': .5,
        'link_distance': 60,
        'friction': .8,
        'charge': -800
    }

    init_viz(settings);

    console.log("Visualization Loaded");
});

// Toggles the settings menu
function toggle_settings() {
    $("#settings-menu").toggle(300, function() {
        if ($("#settings-menu").is(":visible")) {
            $("#toggle-settings").css("background-color", "#666");
            $("#toggle-settings").css("border-top-right-radius", "0px");
            $("#toggle-settings").css("border-bottom-right-radius", "0px");
        }
        else {
            $("#toggle-settings").css("background-color", "");
            $("#toggle-settings").css("border-top-right-radius", "5px");
            $("#toggle-settings").css("border-bottom-right-radius", "5px");
        }
    });
}

// Gets JSON data and initializes D3 force graph
function init_viz(settings) {

    // Clear old viz
    $("#viz-container > svg").remove();
    simulation = null;
    _graph = null;
    original_graph = null;
    original_id_map = null;
    original_link_map = null;
    original_graph = null;
    graph = null;
    node_container = null;
    link = null;
    linked_nodes = [];

    // Deep copy data for restoration
    var viz_data = jQuery.extend(true, {}, ai_concept_map_data);

    // ID of root node
    ROOT_ID = "machine_learning";
    // Should the root node be centered?
    CENTER_ROOT = settings['center_root'];
    // Strength of links (how easily they can be compressed) between nodes [0, INF]
    LINK_STRENGTH = settings['link_strength'];
    // Distance between nodes [0, INF]
    LINK_DISTANCE = settings['link_distance'];
    // Charge between nodes [-INF, INF]
    CHARGE = settings['charge'];
    // How easily particles are dragged across the screen [0, 1]
    FRICTION = settings['friction'];
    // Node coloring scheme. Possible values:
    // "DISTANCE": Color nodes ordinally based on their "distance" attribute using the COLOR_KEY_DISTANCE map
    COLOR_MODE = "DISTANCE";
    // Colors assigned to each distance when COLOR_MODE = "DISTANCE"
    COLOR_KEY_DISTANCE = ["#63D467", "#63B2D4", "#AE63D4", "#D46363", "#ED9A55", "#E5EB7A"];
    // Determines the style of links based on their "type" attribute
    // Values should be an even-length array for alternating black / white segments in px
    LINK_STYLE = {"derivative": "", "related": "10,8"}
    // Method by which the distance from root is calculated. Possible values:
    // "SOURCE": Calculate by traversing source relationships
    // "SHORTEST": Calculate by using Dijkstra's algorithm to find graph-wide shortest distance
    DISTANCE_MODE = "SOURCE";
    // Base node size
    SIZE_BASE = 10;
    // Factor by which to multiply the inverse distance from root in calculating node size
    SIZE_DISTANCE_MULTIPLIER = 25;
    // Factor by which to multiply number of connections in calculating node size
    SIZE_CONNECTIONS_MULTIPLIER = 1.5;
    // Opacity that a node fades to on node hover
    NODE_FADE_OPACITY = .4;
    // Opacity that a link fades to on node hover
    LINK_FADE_OPACITY = .1;
    // Whether to hide nodes with no description
    HIDE_EMPTY_NODES = false;
    // If true, nodes will be collapsed when they are hidden (via the collapsing of a parent node)
    COLLAPSE_HIDDEN = false;

    var graph; // Working copy of graph used in functions
    var original_graph; // Original copy of graph used for restoring nodes
    var link_map; // Maps a node to an array of its links



    // Gets container size
    var outer_container = $("#viz-container");
    var width = outer_container.width();
    var height = outer_container.height();

    // Adds svg box and allows it to resize / zoom as needed
    var svg = d3.select("#viz-container").append("svg")
        .attr("width", "100%")
        .attr("height", "100%")
        .attr("viewBox","0 0 " + Math.min(width, height) + " " + Math.min(width, height))
        .attr("preserveAspectRatio", "xMinYMin")
        .on("contextmenu", container_contextmenu)
        .call(d3.zoom()
            .scaleExtent([.1, 10])
            .on("zoom", container_zoom))
        .on("dblclick.zoom", null); // Don't zoom on double left click

    // Creates actual force graph container (this is what actually gets resized as needed)
    var container = svg.append("g")

    // Initializes force graph simulation
    // NOTE: simulation is global so it can be accessed by outside functions
    // TODO: Create api for upating simulation so we can make this private
    simulation = d3.forceSimulation()
        .force("link", d3.forceLink().id(function(d) { return d.id; }).distance(LINK_DISTANCE).strength(LINK_STRENGTH))
        .force("charge", d3.forceManyBody().strength(CHARGE))
        .force("center", d3.forceCenter(width / 2, height / 2))

    _graph = viz_data;

    // Transfer scope
    original_graph = _graph;

    // QUESTION: Are all of these necessary? Probably not
    original_id_map = generate_id_map(original_graph);
    original_link_map = generate_link_map(original_graph, original_id_map);
    original_graph = calculate_node_distances(original_graph, original_link_map);

    update();
    // });

    // Updates the simulation
    function update() {

        // Recalculate from original to pick up unhidden nodes and in case HIDE_EMPTY_NODES changed
        graph = filter_nodes(original_graph, original_id_map);

        // HACK: Removes old nodes and links
        d3.selectAll(".nodes").remove();
        d3.selectAll(".links").remove();

        // Appends links to container
        var link = container.append("g")
            .attr("class", "links")
            .selectAll("line")
            // Filters out links with a hidden source or target node
            .data(graph.links)
            .enter().append("line")
                .attr("class", "link")
                .attr("stroke-width", 1.5)
                .attr("stroke-dasharray", link_style);

        // Appends nodes to container
        var node_container = container.append("g")
            .attr("class", "nodes")
            .selectAll("g")
            // Filters out hidden nodes and nodes without a description
            .data(graph.nodes)
            .enter().append("g")
            .attr("class", "node")
            .on("mouseover", node_mouseover)
            .on("mouseout", node_mouseout)
            .on("mousedown", node_mousedown)
            .on("click", node_click)
            .on("dblclick", node_dblclick)
            .on("contextmenu", node_contextmenu)
            .call(d3.drag()
                .on("start", container_drag_start)
                .on("drag", container_drag)
                .on("end", container_drag_end));

        // Add node circles
        node_container
            .append("circle")
                .attr("r", node_size)
                .attr("fill", node_color)
                .attr("stroke", node_border_color)
                .attr("stroke-width", node_border_width);

        // Add node labels
        node_container
            .append("text")
                .attr("dx", 12)
                .attr("dy", ".35em")
                .style("color", "#333")
                .text(function(d) { return d.name });

        // Initializes simulation
        simulation
            .nodes(graph.nodes)
            .on("tick", ticked)
            .force("link")
                .links(graph.links);

        // Calculates new node and link positions every tick
        function ticked() {

            if (CENTER_ROOT) {
                node_container
                    .attr("transform", function(d) {
                        if (d.id == ROOT_ID) {
                            d.x = width / 2;
                            d.y = height / 2;
                            d.fx = width / 2;
                            d.fy = height / 2;
                        }
                        return "translate(" + d.x + "," + d.y + ")";
                    });
            }
            else {
                node_container
                    .attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; });
            }

            link
                .attr("x1", function(d) { return d.source.x; })
                .attr("y1", function(d) { return d.source.y; })
                .attr("x2", function(d) { return d.target.x; })
                .attr("y2", function(d) { return d.target.y; });
        }
    }



    // STYLES



    // Sizes nodes
    function node_size(d) {
        return (1 / (d.distance + 1)) * SIZE_DISTANCE_MULTIPLIER + (original_link_map[d.id].length - 1) * SIZE_CONNECTIONS_MULTIPLIER + SIZE_BASE;
    }

    // Color nodes depending on COLOR_MODE
    function node_color(d) {
        if (COLOR_MODE == "DISTANCE") {
            if (d.distance == undefined) return "#333";
            return COLOR_KEY_DISTANCE[d.distance % COLOR_KEY_DISTANCE.length];
        }
        // Default scheme: all dark grey
        return "#333";
    }

    // Colors node borders depending on if they are leaf nodes or not
    function node_border_color(d) {
        // Only one link means it is the target
        if (original_link_map[d.id].filter(function(link) { return link.type == "derivative"; }).length == 1 && d.id != ROOT_ID) return "#333";
        return "#F7F6F2";
    }

    // Draws node borders depending on if they are leaf nodes or not
    function node_border_width(d) {
        // Only one link means it is the target
        if (original_link_map[d.id].length == 1 && d.id != ROOT_ID) return "1.6px";
        return ".8px";
    }

    // Draws links as dash arrays based on their type
    function link_style(d) {
        return LINK_STYLE[d.type];
    }



    // EVENT LISTENERS



    // Node mouseover handler
    function node_mouseover(d) {
        // Create array of linked node ids
        linked_nodes = [];
        for (var i = 0; i < original_link_map[d.id].length; i++) {
            if (linked_nodes.indexOf(original_link_map[d.id][i]["source"].id) == -1) linked_nodes.push(original_link_map[d.id][i]["source"].id);
            if (linked_nodes.indexOf(original_link_map[d.id][i]["target"].id) == -1) linked_nodes.push(original_link_map[d.id][i]["target"].id);
        }
        // Update opacity of all nodes except immediately linked nodes
        d3.selectAll(".node").transition().attr("opacity", function(x) {
            if (linked_nodes.indexOf(x.id) == -1) {
                return NODE_FADE_OPACITY;
            }
            return "1";
        });
        // Update opacity of all links except immediate links
        d3.selectAll(".link").transition().attr("opacity", function(x) {
            for (var i = 0; i < original_link_map[d.id].length; i++) {
                // QUESTION: Can this be simplified?
                // TODO: Some links don't fade when a leaf node is selected
                if (x.source.id != original_link_map[d.id][i].source.id && x.source.id != original_link_map[d.id][i].target.id &&
                    x.target.id != original_link_map[d.id][i].target.id && x.target.id != original_link_map[d.id][i].source.id) return LINK_FADE_OPACITY;
            }
            return "1";
        });
    }

    // Node mouseout handler
    function node_mouseout(d) {
        d3.selectAll(".node").transition().attr("opacity", "1");
        d3.selectAll(".link").transition().attr("opacity", "1");
    }

    // Node mousedown handler
    function node_mousedown(d) {
        // console.log("click");
        // console.log(d.fixed);
        if (d3.event.defaultPrevented) return;
        d3.event.preventDefault(); // Prevent middle click scrolling
        // Unpin node if middle click
        if (d3.event.which == 2) {
            d3.select(this).classed("fixed", d.fixed = false);
            d.fx = null;
            d.fy = null;
        }
        simulation.alpha(.3).restart();
    }

    // Node left click handler
    function node_click(d) {
        if (d3.event.defaultPrevented) return;
        d3.event.preventDefault();
        // Left click: update sidebar
        if (d3.event.which == 1) {
            update_sidebar(d);
        }
    }

    // Node double left click handler
    function node_dblclick(d) {
        // console.log("dblclick");
        // console.log(d.fixed);
        toggle_node(d, true);
        update();
        // Don't pin node if it wasn't before (dblclick triggers click listener, pinning node)
        // TODO: Fix this
        if (!d.fixed) {
            d3.select(this).classed("fixed", d.fixed = false);
            d.fx = null;
            d.fy = null;
        }
    }

    // Node right click handler
    function node_contextmenu(d) {
        // Unpin node
        d3.select(this).classed("fixed", d.fixed = false);
        // HACK: Why doesn't just adding d.fixed = false work?
        d.fx = null;
        d.fy = null;
        simulation.alpha(.3).restart();
    }

    // Container drag start handler
    function container_drag_start(d) {
        // TODO: Why doesn't this prevent the simulation from being restarted when user tries to drag a centered root?
        // NOTE: Maybe it's part of .call(d3.drag()...)?
        if (CENTER_ROOT && d.id == ROOT_ID) return;
        if (!d3.event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
        // Fixes node in place
        d3.select(this).classed("fixed", d.fixed = true);
    }

    // Container drag handler
    function container_drag(d) {
        d.fx = d3.event.x;
        d.fy = d3.event.y;
    }

    // Container dragend handler
    function container_drag_end(d) {
        if (!d3.event.active) simulation.alphaTarget(0);
    }

    // Container right click handler (outside nodes)
    function container_contextmenu(d) {
        d3.event.preventDefault(); // Prevent context menu from appearing
    }

    // Container zoom handler
    function container_zoom() {
        container.attr("transform", d3.event.transform);
    }
}

// Creates dictionary for identifying node objects by their id
// {
//     "id_1": <Node>,
//     ...
// }
function generate_id_map(graph) {
    var id_map = {}
    for (var i = 0; i < graph.nodes.length; i++) {
        id_map[graph.nodes[i].id] = graph.nodes[i];
    }
    return id_map;
}

// Creates dictionary for identifying all links of a certain node
// {
//     "node_1": [
//         {"source": <Node>, "target": <Node>, "type": "..."},
//         ...
//     ],
//     ...}
function generate_link_map(graph, id_map) {
    var link_map = {}
    for (var i = 0; i < graph.nodes.length; i++) {
        link_map[graph.nodes[i].id] = []
        for (var j = 0; j < graph.links.length; j++) {
            if (graph.nodes[i].id == graph.links[j].source || graph.nodes[i].id == graph.links[j].target) {
                // DEPRECATED: This version yields objects of format: {"source": "source_id", "target": "target_id", "type": "..."}
                // link_map[graph.nodes[i].id].push({"source": graph.links[j].source, "target": graph.links[j].target, "type": graph.links[j].type});
                link_map[graph.nodes[i].id].push({"source": id_map[graph.links[j].source], "target": id_map[graph.links[j].target], "type": graph.links[j].type});
            }
        }
    }
    return link_map;
}

// Creates dictionary for identifying shortest distance from the current node to the root node
// {
//     "node_1": 3,
//     ...
// }
function calculate_node_distances(graph, link_map) {
    var distance_map = {}; // Used for memoization
    var distance;
    for (var i = 0; i < graph.nodes.length; i++) {
        if (DISTANCE_MODE == "SOURCE") {
            distance = distance_to_root_source(distance_map, link_map, graph.nodes[i].id);
            graph.nodes[i].distance = distance;
            distance_map[graph.nodes[i].id] = distance;
        }
        else if (DISTANCE_MODE == "SHORTEST") {
            distance = distance_to_root_shortest(graph, graph.nodes[i].id);
            distance_map[graph.nodes[i].id] = distance;
        }
        // console.log(graph.nodes[i].id + ": " + graph.nodes[i].distance);
    }
    return graph;
}

// Calculates the distance from the specified node to the root node by traversing source relations
// NOTE: This only works if the graph has a derivative structure (every node has a path from root)
function distance_to_root_source(distance_map, link_map, id) {
    var cur_id = id;
    var distance = 0;
    var max_distance = 100; // Prevents runaway distances on non-derivative graph structures
    while (cur_id != ROOT_ID) {
        if (distance == max_distance) return max_distance;
        if (distance_map[cur_id] != undefined) return distance_map[cur_id] + distance;
        for (var i = 0; i < link_map[cur_id].length; i++) {
            if (link_map[cur_id][i]["target"].id == cur_id) {
                cur_id = link_map[cur_id][i]["source"].id;
                break;
            }
        }
        distance++;
    }
    return distance;
}

// Calculates the shortest distance from the specified node to the root node using Dijkstra's algorithm
// https://gist.github.com/k-ori/11033337
// TODO: Adapt to graph structure OR scrap and use some memoized version of link_map?
//        *Use distance_map generated from distance_to_root_source()
function distance_to_root_shortest(graph, id) {
    var distance = {},
        prev = {},
        vertices = {},
        u;

    // Setup distance sentinel
    graph.vertex.forEach(function(v_i) {
        distance[v_i] = Infinity;
        prev[v_i] = null;
        vertices[v_i] = true;
    });

    distance[start] = 0;

    while (Object.keys(vertices).length > 0) {
        // Obtain a vertex whose distaance is minimum.
        u = Object.keys(vertices).reduce(function(prev, v_i) {
            return distance[prev] > distance[v_i] ? +v_i : prev;
        }, Object.keys(vertices)[0]);

        graph.edge.filter(function(edge) {
            var from = edge[0],
                to      = edge[1];
            return from===u || to===u;
        })
        .forEach(function(edge) {
            var to = edge[1]===u ? edge[0] : edge[1],
                dist = distance[u] + edge[2];

            if (distance[to] > dist) {
                distance[to] = dist;
                prev[to] = u;
            }
        });
        // Mark visited
        delete vertices[u];
    }
    return distance;
}

function center_root() {
    CENTER_ROOT = !CENTER_ROOT;
    if (CENTER_ROOT) simulation.alpha(.3).restart();
    // simulation.restart();
}

// Filters out empty (if applicable) and hidden nodes and returns the new graph
function filter_nodes(graph, id_map) {
    var filtered_graph = {};
    filtered_graph.links = graph.links.filter(function(link) {
        var link_source = (typeof link.source === "string") ? id_map[link.source] : link.source;
        var link_target = (typeof link.target === "string") ? id_map[link.target] : link.target;
        var empty = HIDE_EMPTY_NODES && (link_source.description == "" || link_target.description == "");
        // Default visibility state of nodes is "shown"
        if (link_source.hidden == undefined) link_source.hidden = "shown";
        if (link_target.hidden == undefined) link_target.hidden = "shown";
        return !empty && (!(link_source.hidden.indexOf("hidden") != -1) && !(link_target.hidden.indexOf("hidden") != -1));
    });
    filtered_graph.nodes = graph.nodes.filter(function(node) {
        var empty = HIDE_EMPTY_NODES && node.description == "";
        // If node has no links, only check if description is empty
        if ("hidden" in node) {
            return !empty && !(node.hidden.indexOf("hidden") != -1);
        }
        else {
            return !empty;
        }
    });
    return filtered_graph;
}

// Toggles all of a node's descendants
function toggle_node(d, top, direction) {
    for (var i = 0; i < original_link_map[d.id].length; i++) {
        // Only hide derivative nodes
        if (original_link_map[d.id][i].type == "derivative" && original_link_map[d.id][i].target.id != d.id) {
            if (COLLAPSE_HIDDEN) {
                if (top) original_link_map[d.id][i].target.hidden = (original_link_map[d.id][i].target.hidden == "hidden") ? "shown" : "hidden";
                else original_link_map[d.id][i].target.hidden = "hidden";
            }
            else {
                var cur_state = original_link_map[d.id][i].target.hidden;
                var new_state;
                if (top) {
                    // Direction is gained from any one of the linked nodes
                    direction = (cur_state.indexOf("hidden") == -1) ? "hide" : "show";
                    new_state = (cur_state.indexOf("hidden") == -1) ? "hidden_shown" : "shown";
                }
                else {
                    if (cur_state == "shown" && direction == "hide") new_state = "hidden_shown";
                    else if (cur_state == "hidden_shown" && direction == "hide") new_state = "hidden_hidden";
                    else if (cur_state == "hidden_shown" && direction == "show") new_state = "shown"
                    else if (cur_state == "hidden_hidden" && direction == "hide") new_state = "hidden_hidden"
                    else if (cur_state == "hidden_hidden" && direction == "show") new_state = "hidden_shown"
                }
                original_link_map[d.id][i].target.hidden = new_state;
            }
            // console.log(original_link_map[d.id][i].target.id + ": " + original_link_map[d.id][i].target.hidden);
            toggle_node(original_link_map[d.id][i].target, false, direction);
        }
    }
}

function api_center_root() {
    console.log("Centering root");
    if ($("#center-root-input").css("background-color") != "rgb(209, 225, 255)") {
        $("#center-root-input").css("background-color", "#D1E1FF");
    }
    else {
        $("#center-root-input").css("background-color", "#777");
    }
    center_root();
}

// Updates one of the settings of the visualization
function api_update_settings(setting, new_val) {
    settings[setting] = new_val;
    init_viz(settings)
}



function update_sidebar(d) {

    $(".warning").addClass("hidden");

    if ($("#concept-name").text() == d.name) return;

    $("#concept-name").text(d.name);

    // "description" section
    var description_elem = $("#concept-description");
    if (d.description != "") {
        if (description_elem.hasClass("hidden")) {
            description_elem.fadeIn(200, function() {
                description_elem.removeClass("hidden");
            });
        }
        description_elem.text(d.description);
    }
    else {
        description_elem.fadeOut(200, function() {
            description_elem.addClass("hidden");
        });
    }

    // "when to use" section
    var when_div = $("#concept-when");
    if (d.when.cases.length != 0 || d.when.description != "") {
        if (when_div.hasClass("hidden")) {
            when_div.fadeIn(200, function() {
                when_div.removeClass("hidden");
            });
        }
        $("#concept-when-description").text(d.when.description);
        var when_list = $("#concept-when-list");
        when_list.empty();
        for (var i = 0; i < d.when.cases.length; i++) {
            when_list.append("<li>" + d.when.cases[i] + "</li>");
        }
    }
    else {
        when_div.fadeOut(200, function() {
            when_div.addClass("hidden");
        });
    }

    // "how to use" section
    var how_div = $("#concept-how");
    if (d.how.steps.length != 0 || d.how.description != "") {
        if (how_div.hasClass("hidden")) {
            how_div.fadeIn(200, function() {
                how_div.removeClass("hidden");
            });
        }
        $("#concept-how-description").text(d.how.description);
        var how_list = $("#concept-how-list");
        how_list.empty();
        for (var i = 0; i < d.how.steps.length; i++) {
            how_list.append("<li>" + d.how.steps[i] + "</li>");
        }
    }
    else {
        how_div.fadeOut(200, function() {
            how_div.addClass("hidden");
        });
    }

    // "tools" section
    var tools_div = $("#concept-tools");
    if (d.tools.links.length != 0 || d.tools.description != "") {
        if (tools_div.hasClass("hidden")) {
            tools_div.fadeIn(200, function() {
                tools_div.removeClass("hidden");
            });
        }
        $("#concept-tools-description").text(d.tools.description);
        var tools_list = $("#concept-tools-list");
        tools_list.empty();
        for (var i = 0; i < d.tools.links.length; i++) {
            tools_list.append("<dt><a href='" + d.tools.links[i].link + "'>" + d.tools.links[i].name + "</a></dt>");
            tools_list.append("<dd>" + d.tools.links[i].description + "</dt>");
        }
    }
    else {
        tools_div.fadeOut(200, function() {
            tools_div.addClass("hidden");
        });
    }

    // "links" section
    var links_div = $("#concept-links");
    if (d.links.links.length != 0 || d.links.description != "") {
        if (links_div.hasClass("hidden")) {
            links_div.fadeIn(200, function() {
                links_div.removeClass("hidden");
            });
        }
        $("#concept-links-description").text(d.links.description);
        var links_list = $("#concept-links-list");
        links_list.empty();
        for (var i = 0; i < d.links.links.length; i++) {
            links_list.append("<dt><a href='" + d.links.links[i].link + "'>" + d.links.links[i].name + "</a></dt>");
            links_list.append("<dd>" + d.links.links[i].description + "</dt>");
        }
    }
    else {
        links_div.fadeOut(200, function() {
            links_div.addClass("hidden");
        });
    }
}
