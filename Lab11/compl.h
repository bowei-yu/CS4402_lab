//
//  complex.c
//  
//
//  Created by stabirca on 14/12/2020.
//

#include <math.h>

typedef struct{double re, im;} complex;

complex compl_prod(complex z1, complex z2){
    
    complex prod;
    
    prod.re = z1.re * z2.re - z1.im * z2.im;
    prod.im = z1.re * z2.im + z1.im * z2.re;
    
    return prod;
}

complex compl_add(complex z1, complex z2){
    
    complex sum;
    
    sum.re = z1.re + z2.re;
    sum.im = z1.im + z2.im;
    
    return sum;
}

double compl_abs(complex z){
    return sqrt(z.re * z.re + z.im * z.im);
}
