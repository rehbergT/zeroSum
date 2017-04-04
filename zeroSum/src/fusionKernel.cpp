#include "fusionKernel.h"


struct fusionKernel* appendElement( struct fusionKernel *preElement, int i, double value) 
{
    struct fusionKernel* element = (struct fusionKernel* )malloc(sizeof(struct fusionKernel));

    element->i     = i;
    element->value = value;
    element->next  = NULL;
    
    if( preElement != NULL )
    {
        struct fusionKernel* tmp = preElement;
        while( tmp->next != NULL) 
            tmp = tmp->next;
        
        tmp->next = element;
        return preElement;
    }
    else
        return element;
}  
