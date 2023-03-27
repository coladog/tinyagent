#include <math.h>
#include "tiny_agent_io.h"

#define GRU_PARAM_MATRIX_NUM 16
#define FC_PARAM_MATRIX_NUM 2
#define TYPE float
#define SIGMOID 0
#define TANH 1
#define RELU 2
#define NONE 3
#define MAXIMUM_MLP_LAYERS 10


struct gru_cell{
    TYPE *wir, *wiz, *win;
    TYPE *whr, *whz, *whn;
    TYPE *bir, *biz, *bin;
    TYPE *bhr, *bhz, *bhn;
    TYPE *rt,  *zt,  *nt, *hidden;
    unsigned int hidden_size, input_size;
    void (* inference)(struct gru_cell *cell, TYPE *input);
    void (* free)(struct gru_cell *cell);
    void (* read)(struct gru_cell *cell, char *path);
    void (* clear)(struct gru_cell *cell);
    // void (* print)(struct gru_cell *cell);
};

struct fc_layer{
    TYPE *w, *b;
    TYPE *result;
    unsigned int input_size, output_size;
    void (* activate_function)(TYPE *vector, unsigned int length);
    void (* read)(struct fc_layer *cell, struct file_holder* holder);
    void (* inference)(struct fc_layer *layer, TYPE *input);
    void (* free)(struct fc_layer *layer);
    void (* print)(struct fc_layer *layer);
};

struct mlp{
    struct fc_layer *layers[MAXIMUM_MLP_LAYERS];
    unsigned int layer_counter;
    TYPE *result;
    void (* add_layer)(struct mlp *nn, struct fc_layer *layer);
    void (* read)(struct mlp *nn, char *path);
    void (* inference)(struct mlp *nn, TYPE *input);
    void (* free)(struct mlp *nn);
    void (* print)(struct mlp *nn);
};

__attribute__((target("sse2"))) void 
matrix_dot_vector(TYPE *matrix, TYPE *vector, int x, int y, TYPE *result, TYPE *bias, int add);

// Activation Function

__attribute__((target("sse2"))) void ta_sigmoid(TYPE *vector, unsigned int length);

__attribute__((target("sse2"))) void ta_tanh(TYPE *vector, unsigned int length);

__attribute__((target("sse2"))) void ta_relu(TYPE *vector, unsigned int length);

// Gated Recurrent Unit (GRU)

__attribute__((target("sse2"))) void inference_gru_cell(struct gru_cell *cell, TYPE *input);

__attribute__((target("sse2"))) void read_in_gru_cell(struct gru_cell *cell, char *path);

__attribute__((target("sse2"))) void init_gru_cell(struct gru_cell *cell, unsigned int hidden_size, unsigned int input_size);

__attribute__((target("sse2"))) void free_gru_cell(struct gru_cell *cell);

__attribute__((target("sse2"))) void clear_gru_hidden(struct gru_cell *cell);

// __attribute__((target("sse2"))) void print_gru(struct gru_cell *cell);

// Fully-Connected Layer (fc layer)

__attribute__((target("sse2"))) void init_fc_layer(struct fc_layer *layer, unsigned int input_size, unsigned int output_size, char activate);

__attribute__((target("sse2"))) void free_fc_layer(struct fc_layer *layer);

__attribute__((target("sse2"))) void inference_fc_layer(struct fc_layer *layer, TYPE *input);

__attribute__((target("sse2"))) void read_in_fc_layer(struct fc_layer *layer, struct file_holder *holder);

// __attribute__((target("sse2"))) void print_fc(struct fc_layer *layer);

// Multi-Layer Perceptron (MLP)

__attribute__((target("sse2"))) void init_mlp(struct mlp *nn);

__attribute__((target("sse2"))) void read_in_mlp(struct mlp *nn, char *path);

__attribute__((target("sse2"))) void add_mlp_layer(struct mlp *nn, struct fc_layer *layer);

__attribute__((target("sse2"))) void inference_mlp(struct mlp *nn, TYPE *input);

__attribute__((target("sse2"))) void free_mlp(struct mlp *nn);

// __attribute__((target("sse2"))) void print_mlp(struct mlp *nn);
