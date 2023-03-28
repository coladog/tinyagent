#include "tiny_agent_io.h"

#define GRU_PARAM_MATRIX_NUM 16
#define FC_PARAM_MATRIX_NUM 2
#define TA_TYPE int
#define SIGMOID 0
#define TANH 1
#define RELU 2
#define NONE 3
#define MAXIMUM_MLP_LAYERS 10

#define F_RANGE 10
#define SCALE_BIT 30
#define TABLE_RANGE 10000

static TA_TYPE *sigmoid_input, *sigmoid_output;
static TA_TYPE *tanh_input, *tanh_output;

struct gru_cell{
    TA_TYPE *wir, *wiz, *win;
    TA_TYPE *whr, *whz, *whn;
    TA_TYPE *bir, *biz, *bin;
    TA_TYPE *bhr, *bhz, *bhn;
    TA_TYPE *rt,  *zt,  *nt, *hidden;
    unsigned int hidden_size, input_size;
    void (* inference)(struct gru_cell *cell, TA_TYPE *input);
    void (* free)(struct gru_cell *cell);
    void (* read)(struct gru_cell *cell, char *path);
    void (* clear)(struct gru_cell *cell);
    // void (* print)(struct gru_cell *cell);
};

struct fc_layer{
    TA_TYPE *w, *b;
    TA_TYPE *result;
    unsigned int input_size, output_size;
    void (* activate_function)(TA_TYPE *vector, unsigned int length);
    void (* read)(struct fc_layer *cell, struct file_holder* holder);
    void (* inference)(struct fc_layer *layer, TA_TYPE *input);
    void (* free)(struct fc_layer *layer);
    // void (* print)(struct fc_layer *layer);
};

struct mlp{
    struct fc_layer *layers[MAXIMUM_MLP_LAYERS];
    unsigned int layer_counter;
    TA_TYPE *result;
    void (* add_layer)(struct mlp *nn, struct fc_layer *layer);
    void (* read)(struct mlp *nn, char *path);
    void (* inference)(struct mlp *nn, TA_TYPE *input);
    void (* free)(struct mlp *nn);
    // void (* print)(struct mlp *nn);
};

void matrix_dot_vector(TA_TYPE *matrix, TA_TYPE *vector, int x, int y, TA_TYPE *result, TA_TYPE *bias, int add);

TA_TYPE q_mul(TA_TYPE x, TA_TYPE y);

TA_TYPE q_add(TA_TYPE x, TA_TYPE y);

void init_sigmoid_table(char *sigmoid_input_file, char *sigmoid_output_file);

void init_tanh_table(char *tanh_input_file, char *tanh_output_file);

void free_sigmoid_table();

void free_tanh_table();

TA_TYPE table_query(TA_TYPE *input, TA_TYPE *output, TA_TYPE v);

TA_TYPE quantized(TA_TYPE v);

// activation function

void ta_sigmoid(TA_TYPE *vector, unsigned int length);

void ta_tanh(TA_TYPE *vector, unsigned int length);

void ta_relu(TA_TYPE *vector, unsigned int length);

// Gated Recurrent Unit (GRU)

void inference_gru_cell(struct gru_cell *cell, TA_TYPE *input);

void read_in_gru_cell(struct gru_cell *cell, char *path);

void init_gru_cell(struct gru_cell *cell, unsigned int hidden_size, unsigned int input_size);

void free_gru_cell(struct gru_cell *cell);

void clear_gru_hidden(struct gru_cell *cell);

// void print_gru(struct gru_cell *cell);

// Fully-connected Layer (fc layer)

void init_fc_layer(struct fc_layer *layer, unsigned int input_size, unsigned int output_size, char activate);

void free_fc_layer(struct fc_layer *layer);

void inference_fc_layer(struct fc_layer *layer, TA_TYPE *input);

void read_in_fc_layer(struct fc_layer *layer, struct file_holder *holder);

// void print_fc(struct fc_layer *layer);

// Multi Layer Perceptron (MLP)

void init_mlp(struct mlp *nn);

void read_in_mlp(struct mlp *nn, char *path);

void add_mlp_layer(struct mlp *nn, struct fc_layer *layer);

void inference_mlp(struct mlp *nn, TA_TYPE *input);

void free_mlp(struct mlp *nn);

// void print_mlp(struct mlp *nn);
