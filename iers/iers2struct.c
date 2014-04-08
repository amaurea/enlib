#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char * addspaces(char * s, int * fieldwidths, int nfield)
{
	int i, j,k1=0,k2=0;
	char * out = malloc(strlen(s)+nfield+1);
	for(i=0;i<nfield;i++)
	{
		for(j=0;j<fieldwidths[i];j++)
			out[k2++] = s[k1++];
		out[k2++] = ' ';
	}
	out[k2] = 0;
	return out;
}

int main()
{
	int year, month, day, mjd, i, mjd0, n;
	double pm[2], epm[2], dUT, edUT, lod, elod, dPsi, edPsi, dEps, edEps, foo[7];
	char ip[3];
	char * line = NULL;
	char * fixed = NULL;
	int fieldwidths[] = {2,2,2,9,2,10,9,10,9,3,10,10,8,7,3,10,9,10,10,11,10,11,10,10};
	size_t len;
	printf("#include <stdlib.h>\n");
	printf("#include <iers.h>\n");
	printf("IERSInfo iers_info[] = {\n");
	for(i=0; getline(&line, &len, stdin) != -1; i++)
	{
		if(fixed) free(fixed);
		fixed = addspaces(line, fieldwidths, sizeof(fieldwidths)/sizeof(int));
		n=sscanf(fixed, "%d %d %d %5d.00 %c %lf %lf %lf %lf %c %lf %lf %lf %lf %c %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
			&year,&month,&day,&mjd,&ip[0],&pm[0],&epm[0],&pm[1],&epm[1],
			&ip[1],&dUT,&edUT,&lod,&elod,&ip[2],&dPsi,&edPsi,&dEps,&edEps,
			&foo[0],&foo[1],&foo[2],&foo[3],&foo[4],&foo[5],&foo[6]);
		if(n < 12) continue;
		year += 1900; if(year < 1950) year += 100;
		printf("	{%4d, %2d, %2d, %5d, '%c', %9.6f, %9.6f, %9.6lf, %9.6lf, '%c', %10.7lf, %10.7lf, %7.4lf, %7.4lf, '%c', %9.3lf, %9.3lf},\n",
			year,month,day,mjd,ip[0],pm[0],epm[0],pm[1],epm[1],ip[1],dUT,edUT,lod,elod,ip[2],dPsi,edPsi,dEps,edEps);
		if(i == 0) mjd0 = mjd;
	}
	printf("};\n");
	printf("#define IERS_MJD0 %d\n", mjd0);
	printf("#define IERS_N %d\n", i);
	printf("IERSInfo * iers_lookup(double mjd) {\n");
	printf("	int idx = (int)(mjd-IERS_MJD0);\n");
	printf("	return idx < 0 ? NULL : idx >= IERS_N ? NULL : &iers_info[idx];\n");
	printf("}\n");
	return 0;
}
