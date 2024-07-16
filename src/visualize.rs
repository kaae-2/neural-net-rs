use std::{collections::HashSet, error::Error};

use graphviz_rust::{
    attributes::NodeAttributes,
    dot_generator::*,
    dot_structures::{Edge, EdgeTy, Graph, Id, Node, NodeId, Stmt, Vertex},
    printer::{DotPrinter, PrinterContext},
};
use uuid::Uuid;

use crate::{engine::Value, neural_net::MLP};

pub(crate) fn visualize_network(network: MLP) -> Result<String, Box<dyn Error>> {
    Ok(String::from("it worked!"))
}

fn trace_nodes(root: Value) -> Result<(HashSet<Value>, HashSet<(Value, Value)>), Box<dyn Error>> {
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
                (nodes, edges) = build(child, nodes, edges).expect("works!");
            }
        }
        Some((nodes, edges))
    }
    (nodes, edges) = build(&root, nodes, edges).expect("works!");

    Ok((nodes, edges))
}

fn uuid_to_id(id: Uuid) -> Result<String, Box<dyn Error>> {
    Ok(id
        .to_string()
        .split(&"-")
        .into_iter()
        .last()
        .expect("values")
        .to_owned())
}

fn draw_dots(root: Value) -> Result<String, Box<dyn Error>> {
    let mut graph = Graph::DiGraph {
        id: Id::Html(String::from("1")),
        strict: true,
        stmts: Vec::new(),
    };
    let (nodes, edges) = trace_nodes(root)?;
    for n in nodes {
        let uid = NodeId(
            Id::Html(format!("\"{}\"", uuid_to_id(n.borrow().uuid)?)),
            None,
        ); // (n.borrow().uuid.to_string());
        graph.add_stmt(Stmt::Node(Node {
            id: uid.clone(),
            attributes: vec![
                NodeAttributes::label(format!(
                    "\" {} | data {:?} | grad {:?} \"",
                    uuid_to_id(n.borrow().uuid)?,
                    n.borrow().data,
                    n.borrow().grad
                )),
                NodeAttributes::shape(graphviz_rust::attributes::shape::rect),
            ],
        }));

        if n.borrow()._op != None {
            let op_id = NodeId(
                Id::Html(format!(
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
                Id::Html(format!("\"{}\"", uuid_to_id(n1.borrow().uuid)?)),
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

            let n2_id = Vertex::N(NodeId(Id::Html(n2_string), None));

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
    use graphviz_rust::{
        cmd::{CommandArg, Format},
        exec_dot,
    };

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
        let L = &d * &f;

        // Call the draw_dots function
        let dot = draw_dots(L).unwrap();

        println!("{}", &dot);

        let format = Format::Png;

        let _graph_svg = exec_dot(
            dot.clone(),
            vec![format.into(), CommandArg::Output("./test.png".to_string())],
        )
        .expect("can generate the dotgraph");

        // Validate the dot output
        assert_eq!(dot, "expected_dot_output");
    }
}
