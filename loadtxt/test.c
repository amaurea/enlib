#include <stdlib.h>
#include <stdio.h>
#include "read_table.h"

int main(int argc, char ** argv)
{
	double * a = NULL;
	int dims[2];
	int status = read_table(argv[1], " \t", "#", &a, dims);
	printf("status: %d %d %d\n", status, dims[0], dims[1]);
#if 0
	if(status)
	{
		int r, c;
		for(r = 0; r < dims[0]; r++)
		{
			for(c = 0; c < dims[1]; c++)
				printf(" %4.1f", a[r*dims[1]+c]);
			printf("\n");
		}
	}
#endif
}
