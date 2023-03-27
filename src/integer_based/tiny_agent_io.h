#include <stdio.h>
#include <stdlib.h>


struct file_holder{
    FILE *ptr;
};

char* assign_memory(unsigned int bytes);

void free_memory(char *ptr);

void open_file(struct file_holder *holder, char *path);

void close_file(struct file_holder *holder);

void reset_holder(struct file_holder *holder);

void read_bytes(struct file_holder *holder, char *buffer, unsigned int bytes);
