use std::collections::HashSet;

use graphviz_rust::{
    attributes::{rankdir::LR, shape::rect, GraphAttributes, NodeAttributes},
    cmd::{CommandArg, Format},
    dot_structures::{Edge, EdgeTy, Graph, Id::Html, Node, NodeId, Stmt, Vertex},
    exec_dot,
    printer::{DotPrinter, PrinterContext},
};
use uuid::Uuid;

use crate::micrograd::{Value, MLP};

use crate::Result;

// use crate::{engine::Value, neural_net::MLP};

pub fn visualize_network(network: MLP, filename: String) -> Result<String> {
    // for multiple roots
    let label_nodes = network.last_layer();
    // TODO: add "label nodes" as root and run function
    println!("{:?}", network);
    let mut total_nodes: HashSet<Value> = HashSet::new();
    let mut total_edges: HashSet<(Value, Value)> = HashSet::new();
    for node in label_nodes {
        let (n, e) = trace_nodes(node)?;
        total_nodes.extend(n);
        total_edges.extend(e);
    }

    let dot = make_graph(total_nodes, total_edges)?;
    let _ = output_graph_file(&dot, filename)?;
    Ok(dot)
}

pub fn output_graph_file(dot_graph: &String, filename: String) -> Result<()> {
    exec_dot(
        dot_graph.clone(),
        vec![Format::Png.into(), CommandArg::Output(filename)],
    )?;
    Ok(())
}

fn trace_nodes(root: Value) -> Result<(HashSet<Value>, HashSet<(Value, Value)>)> {
    let (mut nodes, mut edges): (HashSet<Value>, HashSet<(Value, Value)>) =
        (HashSet::new(), HashSet::new());
    fn build<'a>(
        v: &'a Value,
        mut nodes: HashSet<Value>,
        mut edges: HashSet<(Value, Value)>,
    ) -> Option<(HashSet<Value>, HashSet<(Value, Value)>)> {
        if !nodes.contains(v) {
            nodes.insert(v.clone());
            for child in &v.borrow()._prev {
                edges.insert((child.clone(), v.clone()));
                (nodes, edges) = build(child, nodes, edges)?;
            }
        }
        Some((nodes, edges))
    }
    (nodes, edges) = build(&root, nodes, edges).expect("works!");

    Ok((nodes, edges))
}

fn uuid_to_id(id: Uuid) -> Result<String> {
    Ok(id
        .to_string()
        .split(&"-")
        .into_iter()
        .last()
        .expect("values")
        .to_owned())
}

pub fn draw_dots(root: Value) -> Result<String> {
    // for a single root
    let (nodes, edges) = trace_nodes(root)?;
    make_graph(nodes, edges)
}

fn make_graph(nodes: HashSet<Value>, edges: HashSet<(Value, Value)>) -> Result<String> {
    let mut graph = Graph::DiGraph {
        id: Html(String::from("1")),
        strict: true,
        stmts: Vec::new(),
    };
    graph.add_stmt(Stmt::Attribute(GraphAttributes::rankdir(LR)));
    // merge nodes in list)

    for n in nodes {
        let uid = NodeId(Html(format!("\"{}\"", uuid_to_id(n.borrow().uuid)?)), None);
        graph.add_stmt(Stmt::Node(Node {
            id: uid.clone(),
            attributes: vec![
                NodeAttributes::label(format!(
                    "\" {} | data {:?} | grad {:?} \"",
                    uuid_to_id(n.borrow().uuid)?,
                    n.borrow().data,
                    n.borrow().grad
                )),
                NodeAttributes::shape(rect),
            ],
        }));

        if n.borrow()._op != None {
            let op_id = NodeId(
                Html(format!(
                    "\"{}_{:?}\"",
                    uuid_to_id(n.borrow().uuid)?,
                    n.borrow()._op.as_ref().unwrap()
                )),
                None,
            );
            graph.add_stmt(Stmt::Node(Node {
                id: op_id.to_owned(),
                attributes: vec![NodeAttributes::label(format!(
                    "{:?}",
                    n.borrow()._op.as_ref().unwrap()
                ))],
            }));
            graph.add_stmt(Stmt::Edge(Edge {
                ty: EdgeTy::Pair(Vertex::N(op_id), Vertex::N(uid)),
                attributes: vec![],
            }))
        }
        for (n1, n2) in &edges {
            let n1_id = Vertex::N(NodeId(
                Html(format!("\"{}\"", uuid_to_id(n1.borrow().uuid)?)),
                None,
            ));
            let n2_string = if n2.borrow()._op.as_ref() != None {
                format!(
                    "\"{}_{:?}\"",
                    uuid_to_id(n2.borrow().uuid)?,
                    n2.borrow()._op.as_ref().unwrap()
                )
            } else {
                format!("\"{}\"", uuid_to_id(n2.borrow().uuid)?)
            };

            let n2_id = Vertex::N(NodeId(Html(n2_string), None));

            graph.add_stmt(Stmt::Edge(Edge {
                ty: EdgeTy::Pair(n1_id, n2_id),
                attributes: vec![],
            }))
        }
    }

    let dot = graph.print(&mut PrinterContext::default());
    Ok(dot)
}
#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_draw_dots() {
        // Create a set of nodes for testing
        let a = Value::from(2.0);
        let b = Value::from(-3.0);
        let c = Value::from(10.0);
        let e = &a * &b;
        let d = &e + &c;
        let f = Value::from(-2.0);
        let label = &d * &f;

        // Call the draw_dots function
        let dot = draw_dots(label).unwrap();

        println!("{}", &dot);

        // Validate the dot output
        assert_eq!(dot, "expected_dot_output");
    }
}
