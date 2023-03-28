#include "tiny_agent_io.h"


char* assign_memory(unsigned int bytes)
{
    return (char *)kmalloc(bytes, GFP_KERNEL);
}

void free_memory(char *ptr)
{
    kfree(ptr);
}

void open_file(struct file_holder *holder, char *path)
{
	holder->file = filp_open(path, O_RDWR | O_CREAT, 0644);
	holder->pos = 0;
}

void close_file(struct file_holder *holder)
{
	filp_close(holder->file, NULL);
}

void reset_holder(struct file_holder *holder)
{
	holder->pos = 0;
}

void read_bytes(struct file_holder *holder, char *buffer, unsigned int bytes)
{
	holder->old_fs = get_fs();
    set_fs(KERNEL_DS);
	vfs_read(holder->file, buffer, bytes, &holder->pos);
    set_fs(holder->old_fs);
}

void write_bytes(struct file_holder *holder, char *buffer, unsigned int bytes)
{
	holder->old_fs = get_fs();
    set_fs(KERNEL_DS);
	vfs_write(holder->file, buffer, bytes, &holder->pos);
    set_fs(holder->old_fs);
}
