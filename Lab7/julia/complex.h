typedef struct {
        double re;
        double im;
} Comp;

Comp CompAdd(Comp a, Comp b);
Comp CompProd(Comp a, Comp b);
Comp CompProd2(Comp a, Comp b);
Comp CompConj(Comp a);
double CompAbs(Comp a);
