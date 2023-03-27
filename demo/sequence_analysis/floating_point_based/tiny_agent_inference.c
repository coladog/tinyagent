#include "../../../src/floating_point_based/tiny_agent_f.h"

#define GRU_HIDDEN_SIZE 4
#define GRU_INPUT_SIZE 1
#define FC1_INPUT_SIZE 4
#define FC2_INPUT_SIZE 4
#define PREDICTION_SIZE 2

#define GRU_LOAD_PATH "./data/gru.ta"
#define MLP_LOAD_PATH "./data/mlp.ta"
#define MAX_SEQ_LENGTH 20


int main()
{
    int i, j;

    // # don't generate sequence length > MAX_SEQ_LENGTH
    float input_0[] = {1, 0, 1, 1, 0, 0, 1, 0, 1};
    float input_1[] = {0, 0, 1, 0, 1};
    float input_2[] = {1, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    float *data_to_test[] = {input_0, input_1, input_2};
    int data_to_test_len = 3;
    int seq_len[] = {9, 5, 10};

    struct gru_cell cell;
    init_gru_cell(&cell, GRU_HIDDEN_SIZE, GRU_INPUT_SIZE);
    cell.read(&cell, GRU_LOAD_PATH);

    struct fc_layer layer1, layer2;
    init_fc_layer(&layer1, FC1_INPUT_SIZE, FC2_INPUT_SIZE, SIGMOID);
    init_fc_layer(&layer2, FC2_INPUT_SIZE, PREDICTION_SIZE, NONE);

    struct mlp nn;
    init_mlp(&nn);
    nn.add_layer(&nn, &layer1);
    nn.add_layer(&nn, &layer2);
    nn.read(&nn, MLP_LOAD_PATH);

    printf("[+] read param from:\n");
    printf("[+] \t %s\n", GRU_LOAD_PATH);
    printf("[+] \t %s\n", MLP_LOAD_PATH);
    printf("[+]\n");

    for (i = 0; i < data_to_test_len; i++) {
        float *input = data_to_test[i];

        cell.clear(&cell);
        for (j = 0; j < seq_len[i]; j++) {
            cell.inference(&cell, input + j);
        }
        nn.inference(&nn, cell.hidden);

        printf("[+] now testing sequence: [");
        for (j = 0; j < seq_len[i]; j++) {
            printf("%d", (int) input[j]);
            if (j != seq_len[i] - 1) {
                printf(", ");
            }
        }
        printf("]\n");
        printf("[+] \t prediction: [%.4f, %.4f] -> number of 1: %.0f | length of sequence: %.0f \n", 
            nn.result[0], nn.result[1], (nn.result[0] * seq_len[i]), (nn.result[1] * MAX_SEQ_LENGTH));
    }

    cell.free(&cell);
    nn.free(&nn);
}
