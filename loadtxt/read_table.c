#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "read_table.h"

typedef struct { FILE * f; char buf[0x1000]; int i, n, end; } Fbuf;
int eat_delim(Fbuf * f, char * delim);
int eat_newline(Fbuf * f);
int eat_line(Fbuf * f);
int eat_comment(Fbuf * f, char * comments);
int eat_record(Fbuf * f, char * delim, char ** record);
int read_record(Fbuf * f, char * delim, double * record);

#if 1
int bgetc(Fbuf * f) {
	if(f->i >= f->n) {
		if(f->end) return EOF;
		f->n = fread(f->buf, 1, sizeof(f->buf), f->f);
		f->i = 0;
		if(f->n < sizeof(f->buf)) f->end = 1;
	}
	return f->buf[f->i++];
}
int bungetc(int c, Fbuf * f) { return f->buf[--f->i] = c; }
#else
int bgetc(Fbuf * f) { return fgetc(f->f); }
int bungetc(int c, Fbuf * f) { return ungetc(c, f->f); }
#endif

enum { OK, BAD, END };
int read_table(char * filename, char * delim, char * comments, double ** arr, int * dims)
{
	size_t ncol = 0, nrow = 0, col, n = 0, size = 0x1000, stat;
	double val;
	FILE * file = fopen(filename, "r");
	Fbuf fbuf = { file, "", 0, 0, 0 };
	Fbuf * f = &fbuf;
	if(!f) return 1;
	*arr = malloc(size*sizeof(double));
	for(;;)
	{
		while(eat_comment(f, comments));
		for(col=0;(stat=read_record(f, delim, &val))==OK;col++,n++)
		{
			if(n >= size)
			{
				size *= 2;
				*arr = realloc(*arr, size*sizeof(double));
			}
			(*arr)[n] = val;
		}
		if(stat == BAD) { fclose(f->f); return 0; }
		/* Ignore empty lines */
		if(col)
		{
			if(!ncol) ncol = col;
			else if(ncol != col)
			{
				/* Variable number of columns not allowed */
				free(*arr); fclose(f->f); return 0;
			}
			++nrow;
		}
		/* Only count non-empty rows */
		if(!eat_newline(f))
		{
			dims[0] = nrow;
			dims[1] = ncol;
			return 1;
		}
	}
}

int eat_delim(Fbuf * f, char * delim)
{
	int c, n;
	for(n = 0;;n++)
	{
		c = bgetc(f);
		if(c == EOF) return n;
		if(!strchr(delim,c))
		{
			bungetc(c, f);
			return n;
		}
	}
}

int eat_newline(Fbuf * f)
{
	int c = bgetc(f);
	if(c == EOF) return 0;
	if(c == '\n') return 1;
	bungetc(c, f);
	return 0;
}

int eat_line(Fbuf * f)
{
	int c, n;
	for(n=0;;n++)
	{
		c = bgetc(f);
		if(c=='\n' || c==EOF) return n;
	}
}

int eat_comment(Fbuf * f, char * comments)
{
	int c, n;
	c = bgetc(f);
	if(c == EOF) return 0;
	if(strchr(comments, c)) return eat_line(f);
	bungetc(c, f);
	return 0;
}

int eat_record(Fbuf * f, char * delim, char ** record)
{
	int c, len = 0x100, n;
	*record = realloc(*record, len);
	for(n=0;;n++)
	{
		c = bgetc(f);
		if(c == EOF) break;
		if(strchr(delim, c) || c=='\n')
		{
			bungetc(c, f);
			break;
		}
		if(n >= len)
		{
			len *= 2;
			*record = realloc(*record, len);
		}
		(*record)[n] = (char)c;
	}
	(*record)[n] = '\0';
	return n;
}

int read_record(Fbuf * f, char * delim, double * record)
{
	int n;
	char * word = NULL, * endptr;
	n = eat_delim(f, delim);
	if(n == EOF) return END;
	n = eat_record(f, delim, &word);
	if(n > 0)
	{
		*record = strtod(word, &endptr);
		free(word);
		return endptr != word+n ? BAD : OK;
	}
	if(word != NULL) free(word);
	return END;
}
