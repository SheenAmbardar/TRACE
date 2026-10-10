
import networkx as nx


# ==========================================================
# SECTION 1: DEFINE THE MICROSERVICE DEPENDENCY GRAPH
# ==========================================================

def build_service_graph():
    """Build the directed graph of service dependencies."""

    graph = nx.DiGraph()

    services = ["api", "order", "payment", "inventory"]

    graph.add_nodes_from(services)

    # An edge A -> B means service A depends on service B.
    graph.add_edge("api", "order")
    graph.add_edge("order", "payment")
    graph.add_edge("order", "inventory")

    return graph


# ==========================================================
# SECTION 2: FIND SERVICES AFFECTED BY A DEPENDENCY
# ==========================================================

def get_downstream_services(graph, service):
    """Return services that depend on the given service, directly or indirectly."""

    if service not in graph:
        raise ValueError(f"Unknown service: {service}")

    affected = []

    # Reverse the graph because edges point from caller to dependency.
    reverse_graph = graph.reverse(copy=False)

    for candidate in graph.nodes:
        if candidate == service:
            continue

        if nx.has_path(reverse_graph, service, candidate):
            affected.append(candidate)

    return affected


# ==========================================================
# SECTION 3: DISPLAY THE GRAPH STRUCTURE
# ==========================================================

def describe_service_graph(graph):
    """Print each service and its direct dependencies."""

    print("TRACE service dependency graph")

    for service in graph.nodes:
        dependencies = list(graph.successors(service))

        if dependencies:
            print(f"{service} depends on: {', '.join(dependencies)}")
        else:
            print(f"{service} has no recorded dependencies")


# ==========================================================
# SECTION 4: RUN A SMALL GRAPH TEST
# ==========================================================

if __name__ == "__main__":
    service_graph = build_service_graph()

    describe_service_graph(service_graph)

    print(
        "\nServices potentially affected if payment fails:",
        get_downstream_services(service_graph, "payment"),
    )
