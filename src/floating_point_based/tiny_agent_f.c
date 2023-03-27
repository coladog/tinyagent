#include "tiny_agent_f.h"


__attribute__((target("sse2"))) void 
matrix_dot_vector(TYPE *matrix, TYPE *vector, int x, int y, TYPE *result, TYPE *bias, int add)
{
	int i, j;

	for (i = 0; i < x; i++) {
		if (add)
			result[i] += bias[i];
		else
			result[i] = bias[i];

		for (j = 0; j < y; j++) {
			result[i] += vector[j] * (*(matrix + i * y + j));
		}
	}
}

__attribute__((target("sse2"))) void ta_sigmoid(TYPE *vector, unsigned int length)
{
	while (length--)
		vector[length] = 1.0f / (1.0f + exp(-vector[length]));
}

__attribute__((target("sse2"))) void ta_tanh(TYPE *vector, unsigned int length)
{
	TYPE x, y;
	while (length) {
		x = exp(vector[--length]);
		y = exp(-vector[length]);
		vector[length] = (x - y) / (x + y);
	}
}

__attribute__((target("sse2"))) void ta_relu(TYPE *vector, unsigned int length)
{
	while (length)
		if (vector[--length] < 0)
			vector[length] = 0;
}

__attribute__((target("sse2"))) void inference_gru_cell(struct gru_cell *cell, TYPE *input)
{
    int i;
	TYPE *hidden = cell->hidden;
	matrix_dot_vector(cell->wir, input, cell->hidden_size, cell->input_size, cell->rt, cell->bir, 0);
	matrix_dot_vector(cell->whr, hidden, cell->hidden_size, cell->hidden_size, cell->rt, cell->bhr, 1);
	ta_sigmoid(cell->rt, cell->hidden_size);
	matrix_dot_vector(cell->wiz, input, cell->hidden_size, cell->input_size, cell->zt, cell->biz, 0);
	matrix_dot_vector(cell->whz, hidden, cell->hidden_size, cell->hidden_size, cell->zt, cell->bhz, 1);
	ta_sigmoid(cell->zt, cell->hidden_size);
	matrix_dot_vector(cell->whn, hidden, cell->hidden_size, cell->hidden_size, cell->nt, cell->bhn, 0);
	for (i = 0; i < cell->hidden_size; i++)
		cell->nt[i] *= cell->rt[i];
	matrix_dot_vector(cell->win, input, cell->hidden_size, cell->input_size, cell->nt, cell->bin, 1);
	ta_tanh(cell->nt, cell->hidden_size);
	for (i = 0; i < cell->hidden_size; i++) {
		hidden[i] = (1 - cell->zt[i]) * cell->nt[i] + cell->zt[i] * hidden[i];
	}
}

__attribute__((target("sse2"))) void clear_gru_hidden(struct gru_cell *cell)
{
	int i;
	for (i = 0; i < cell->hidden_size; i++)
		cell->hidden[i] = 0;
}

// __attribute__((target("sse2"))) void print_gru(struct gru_cell *cell)
// {
// 	int i;
// 	printf("hidden: ");
// 	for (i = 0; i < cell->hidden_size; i++)
// 		printf("%.4f ", cell->hidden[i]);
// 	printf("\n");
// }

__attribute__((target("sse2"))) void init_gru_cell(struct gru_cell *cell, unsigned int hidden_size, unsigned int input_size)
{
	int i;
	int size[] = {
			hidden_size * input_size,
			hidden_size * input_size,
			hidden_size * input_size,
			hidden_size * hidden_size,
			hidden_size * hidden_size,
			hidden_size * hidden_size,
			hidden_size,
			hidden_size,
			hidden_size,
			hidden_size,
			hidden_size,
			hidden_size,
			hidden_size,
			hidden_size,
			hidden_size,
			hidden_size,
	};
	TYPE **parameter_matrix[] = {
			&cell->wir,
			&cell->wiz,
			&cell->win,
			&cell->whr,
			&cell->whz,
			&cell->whn,
			&cell->bir,
			&cell->biz,
			&cell->bin,
			&cell->bhr,
			&cell->bhz,
			&cell->bhn,
			&cell->rt,
			&cell->zt,
			&cell->nt,
			&cell->hidden,
	};
	cell->hidden_size = hidden_size, cell->input_size = input_size;

	for (i = 0; i < GRU_PARAM_MATRIX_NUM; i++) 
		*parameter_matrix[i] = (TYPE *)assign_memory(size[i] * sizeof(TYPE));
	
	cell->inference = &inference_gru_cell;
	cell->free = &free_gru_cell;
	cell->read = &read_in_gru_cell;
	cell->clear = &clear_gru_hidden;
	// cell->print = &print_gru;
	cell->clear(cell);
}

__attribute__((target("sse2"))) void free_gru_cell(struct gru_cell *cell)
{
	int i;
	TYPE **parameter_matrix[] = {
			&cell->wir,
			&cell->wiz,
			&cell->win,
			&cell->whr,
			&cell->whz,
			&cell->whn,
			&cell->bir,
			&cell->biz,
			&cell->bin,
			&cell->bhr,
			&cell->bhz,
			&cell->bhn,
			&cell->rt,
			&cell->zt,
			&cell->nt,
			&cell->hidden,
	};

	for (i = 0; i < GRU_PARAM_MATRIX_NUM; i++) 
		free_memory((char*) *parameter_matrix[i]);
}

__attribute__((target("sse2"))) void read_in_gru_cell(struct gru_cell *cell, char *path)
{
	int i;
	int hidden_size = cell->hidden_size, input_size = cell->input_size;
	int size[] = {
			hidden_size * input_size,
			hidden_size * input_size,
			hidden_size * input_size,
			hidden_size * hidden_size,
			hidden_size * hidden_size,
			hidden_size * hidden_size,
			hidden_size,
			hidden_size,
			hidden_size,
			hidden_size,
			hidden_size,
			hidden_size,
			hidden_size,
			hidden_size,
			hidden_size,
	};
	TYPE **parameter_matrix[] = {
			&cell->wir,
			&cell->wiz,
			&cell->win,
			&cell->whr,
			&cell->whz,
			&cell->whn,
			&cell->bir,
			&cell->biz,
			&cell->bin,
			&cell->bhr,
			&cell->bhz,
			&cell->bhn,
			&cell->rt,
			&cell->zt,
			&cell->nt,
	};
	struct file_holder holder;

	open_file(&holder, path);
	for (i = 0; i < GRU_PARAM_MATRIX_NUM - 1; i++) {
		read_bytes(&holder, (char *) *parameter_matrix[i], size[i] * sizeof(TYPE));
	}
	close_file(&holder);
}

// __attribute__((target("sse2"))) void print_fc(struct fc_layer *layer)
// {
// 	int i;
// 	for (i = 0; i < layer->output_size; i++)
// 		printf("%.4f ", layer->result[i]);
// 	printf("\n");
// }

__attribute__((target("sse2"))) void init_fc_layer(struct fc_layer *layer, unsigned int input_size, unsigned int output_size, char activate)
{
	layer->input_size = input_size;
	layer->output_size = output_size;
	layer->result = (TYPE *)assign_memory(output_size * sizeof(TYPE));
	layer->b = (TYPE *)assign_memory(output_size * sizeof(TYPE));
	layer->w = (TYPE *)assign_memory(input_size * output_size * sizeof(TYPE));

	switch (activate)
	{
	case SIGMOID:
		layer->activate_function = &ta_sigmoid;
		break;
	case TANH:
		layer->activate_function = &ta_tanh;
		break;
	case RELU:
		layer->activate_function = &ta_relu;
		break;
	default:
		layer->activate_function = NULL;
		break;
	}

	layer->read = &read_in_fc_layer;
	layer->inference = &inference_fc_layer;
	layer->free = &free_fc_layer;
	// layer->print = &print_fc;
}

__attribute__((target("sse2"))) void free_fc_layer(struct fc_layer *layer)
{
	free_memory((char*) layer->result);
	free_memory((char*) layer->b);
	free_memory((char*) layer->w);
}

__attribute__((target("sse2"))) void inference_fc_layer(struct fc_layer *layer, TYPE *input)
{
	matrix_dot_vector(layer->w, input, layer->output_size, layer->input_size, layer->result, layer->b, 0);
	if (layer->activate_function)
		layer->activate_function(layer->result, layer->output_size);
}

__attribute__((target("sse2"))) void read_in_fc_layer(struct fc_layer *layer, struct file_holder *holder)
{
	int i;
	int input_size = layer->input_size, output_size = layer->output_size;
	int size[] = {
			input_size * output_size,
			output_size
	};
	TYPE **parameter_matrix[] = {
			&layer->w,
			&layer->b
	};

	for (i = 0; i < FC_PARAM_MATRIX_NUM; i++) {
		read_bytes(holder, (char *) *parameter_matrix[i], size[i] * sizeof(TYPE));
	}
}

__attribute__((target("sse2"))) void init_mlp(struct mlp *nn) 
{
	int i;
	for (i = 0; i < MAXIMUM_MLP_LAYERS; i++)
		nn->layers[i] = NULL;

	nn->result = NULL;
	nn->add_layer = &add_mlp_layer;
	nn->inference = &inference_mlp;
	nn->read = &read_in_mlp;
	nn->layer_counter = 0;
	nn->free = free_mlp;
	// nn->print = print_mlp;
}

__attribute__((target("sse2"))) void add_mlp_layer(struct mlp *nn, struct fc_layer *layer)
{
	nn->layers[nn->layer_counter++] = layer;
	nn->result = layer->result;
}

__attribute__((target("sse2"))) void read_in_mlp(struct mlp *nn, char *path)
{
	int i;
	struct file_holder holder;
	open_file(&holder, path);
	
	for (i = 0; i < nn->layer_counter; i++)
		nn->layers[i]->read(nn->layers[i], &holder);

	close_file(&holder);
}

__attribute__((target("sse2"))) void inference_mlp(struct mlp *nn, TYPE *input)
{
	int i;
	for (i = 0; i < nn->layer_counter; i++) {
		nn->layers[i]->inference(nn->layers[i], input);
		input = nn->layers[i]->result;
	}
}

__attribute__((target("sse2"))) void free_mlp(struct mlp *nn)
{
	int i;
	for (i = 0; i < nn->layer_counter; i++)
		nn->layers[i]->free(nn->layers[i]);
}

// __attribute__((target("sse2"))) void print_mlp(struct mlp *nn)
// {
// 	nn->layers[nn->layer_counter - 1]->print(nn->layers[nn->layer_counter - 1]);
// }
 
