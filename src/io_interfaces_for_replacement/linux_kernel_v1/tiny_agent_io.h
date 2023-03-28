#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/slab.h>

struct file_holder{
	struct file *file;
	loff_t pos;
	mm_segment_t old_fs;
};

char* assign_memory(unsigned int bytes);

void free_memory(char *ptr);

void open_file(struct file_holder *holder, char *path);

void close_file(struct file_holder *holder);

void reset_holder(struct file_holder *holder);

void read_bytes(struct file_holder *holder, char *buffer, unsigned int bytes);

void write_bytes(struct file_holder *holder, char *buffer, unsigned int bytes);

