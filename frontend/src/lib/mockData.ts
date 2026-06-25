export const mockGraphData = {
  nodes: [
    { id: "1905.12345", title: "Foundations of QEC", role: "Structural Bottleneck", val: 5 },
    { id: "2401.98765", title: "Surface Codes in Qubits", role: "Semantic Hit", val: 3 },
    { id: "2104.55555", title: "Lattice Surgery", role: "Semantic Hit", val: 2 },
    { id: "1212.44444", title: "Topological Quantum Memory", role: "Structural Bottleneck", val: 4 }
  ],
  links: [
    { source: "2401.98765", target: "1905.12345", weight: 2 },
    { source: "2104.55555", target: "1905.12345", weight: 1.5 },
    { source: "2104.55555", target: "1212.44444", weight: 3 },
    { source: "2401.98765", target: "1212.44444", weight: 1.2 }
  ]
};

export const mockTreeData: Record<string, any> = {
  "1905.12345": [
    { title: "Introduction", content: [{ type: "text", text: "Quantum error correction (QEC) is fundamental..." }] },
    { title: "Theoretical Framework", content: [
      { type: "text", text: "The stabilizer formalism defines..." },
      { type: "equation", text: "S_i |\\psi\\rangle = |\\psi\\rangle \\quad \\forall i" }
    ]}
  ],
  "2401.98765": [
    { title: "Experimental Setup", content: [{ type: "text", text: "We apply the stabilizer framework to 2D grids..." }] },
    { title: "Methodology", content: [{ type: "equation", text: "H = -J \\sum_{<i,j>} Z_i Z_j" }] }
  ]
};
