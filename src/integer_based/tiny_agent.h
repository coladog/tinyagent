#include "tiny_agent_io.h"

#define GRU_PARAM_MATRIX_NUM 16
#define FC_PARAM_MATRIX_NUM 2
#define TYPE int
#define SIGMOID 0
#define TANH 1
#define RELU 2
#define NONE 3
#define MAXIMUM_MLP_LAYERS 10

#define F_RANGE 10
#define SCALE_BIT 30
#define TABLE_RANGE 10000

static TYPE *sigmoid_input, *sigmoid_output;
static TYPE *tanh_input, *tanh_output;

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
    // void (* print)(struct fc_layer *layer);
};

struct mlp{
    struct fc_layer *layers[MAXIMUM_MLP_LAYERS];
    unsigned int layer_counter;
    TYPE *result;
    void (* add_layer)(struct mlp *nn, struct fc_layer *layer);
    void (* read)(struct mlp *nn, char *path);
    void (* inference)(struct mlp *nn, TYPE *input);
    void (* free)(struct mlp *nn);
    // void (* print)(struct mlp *nn);
};

void matrix_dot_vector(TYPE *matrix, TYPE *vector, int x, int y, TYPE *result, TYPE *bias, int add);

TYPE q_mul(TYPE x, TYPE y);

TYPE q_add(TYPE x, TYPE y);

void init_sigmoid_table(char *sigmoid_input_file, char *sigmoid_output_file);

void init_tanh_table(char *tanh_input_file, char *tanh_output_file);

void free_sigmoid_table();

void free_tanh_table();

TYPE table_query(TYPE *input, TYPE *output, TYPE v);

TYPE quantized(TYPE v);

// activation function

void ta_sigmoid(TYPE *vector, unsigned int length);

void ta_tanh(TYPE *vector, unsigned int length);

void ta_relu(TYPE *vector, unsigned int length);

// Gated Recurrent Unit (GRU)

void inference_gru_cell(struct gru_cell *cell, TYPE *input);

void read_in_gru_cell(struct gru_cell *cell, char *path);

void init_gru_cell(struct gru_cell *cell, unsigned int hidden_size, unsigned int input_size);

void free_gru_cell(struct gru_cell *cell);

void clear_gru_hidden(struct gru_cell *cell);

// void print_gru(struct gru_cell *cell);

// Fully-connected Layer (fc layer)

void init_fc_layer(struct fc_layer *layer, unsigned int input_size, unsigned int output_size, char activate);

void free_fc_layer(struct fc_layer *layer);

void inference_fc_layer(struct fc_layer *layer, TYPE *input);

void read_in_fc_layer(struct fc_layer *layer, struct file_holder *holder);

// void print_fc(struct fc_layer *layer);

// Multi Layer Perceptron (MLP)

void init_mlp(struct mlp *nn);

void read_in_mlp(struct mlp *nn, char *path);

void add_mlp_layer(struct mlp *nn, struct fc_layer *layer);

void inference_mlp(struct mlp *nn, TYPE *input);

void free_mlp(struct mlp *nn);

// void print_mlp(struct mlp *nn);
