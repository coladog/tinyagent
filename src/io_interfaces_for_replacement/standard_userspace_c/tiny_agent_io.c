#include "tiny_agent_io.h"


char* assign_memory(unsigned int bytes)
{
    return (char *)malloc(bytes);
}

void free_memory(char *ptr)
{
    free(ptr);
}

void open_file(struct file_holder *holder, char *path)
{
    holder->ptr = fopen(path, "rb");
}

void close_file(struct file_holder *holder)
{
    fclose(holder->ptr);
}

void reset_holder(struct file_holder *holder)
{
    fseek(holder->ptr, 0, SEEK_SET);
}

void read_bytes(struct file_holder *holder, char *buffer, unsigned int bytes)
{
    fread(buffer, bytes, 1, holder->ptr);
}
