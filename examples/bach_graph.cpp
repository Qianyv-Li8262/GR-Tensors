// Compile the existing full Bach expression into a type-level execution DAG.
// Central2 by default; define BACH_CENTRAL4 for the larger stencil variant.
// Running the executable prints graph metadata without numerical evaluation.
#define BACH_COMPILE_GRAPH
#include "bach_tensor.cpp"
