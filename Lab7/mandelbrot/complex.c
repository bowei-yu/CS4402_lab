#include <math.h>
#include "complex.h"

Comp CompAdd(Comp a, Comp b)
{
	Comp c;
	c.re = a.re + b.re;
	c.im = a.im + b.im;
	return c;
}

Comp CompProd(Comp a, Comp b)
{
	Comp c;
	float temp1, temp2;
	temp1 = a.re * b.re;
	temp2 = a.im * b.im;
	c.re = temp1 - temp2;
	c.im = (a.re + a.im) * (b.re + b.im) - (temp1 + temp2);
	return c;
}

Comp CompProd2(Comp a, Comp b)
{
	Comp c;
	c.re = a.re*b.re - a.im*b.im;
	c.im = a.im*b.re + b.im*a.re;
	return c;
}

Comp CompConj(Comp a)
{
	Comp c;
	c.re = a.re;
	c.im = -a.im;
	return c;
}

double CompAbs(Comp a)
{
	double res = sqrt((a.re*a.re) + (a.im*a.im));
	return res;	
}
/*
int main(int argc, char* argv[]){
	Comp a, b, c;
	double res;
	
	printf("Test\n");
	a.re = 314.3;
	a.im = 381.18;
	b.re = 238.029;
	b.im = 91.92;
	printf("a = %lf %lf\n",a.re,a.im);
	printf("b = %lf %lf\n",b.re,b.im);
	c = CompAdd (a,b);
	printf("Addition: %lf + %lf\n",c.re,c.im);	
	printf("Abs a: %lf \n",CompAbs(a));
	c = CompProd (a,b);	
	printf("Product: %lf, %lf\n",c.re,c.im);
	c = CompConj(a);
	printf("conjugate a: %lf, %lf\n",c.re,c.im);
	c = CompConj(c);
	printf("conjugate conjugate: %lf, %lf\n",c.re,c.im);
}
*/
