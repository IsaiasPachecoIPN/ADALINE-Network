%RNA ADALINE - Pacheco Castillo Isaias

%Conjunto de datos de entrenamiento
archivoDataSet=input('Nombre del conjunto de datos de entrenamiento: ','s')
conjuntoDeEntrenamiento = load(archivoDataSet)

%Valores objetivo del conjunto de datos de entrenamiento
archivoTarget = input('Nombre del archivo con los objetivos: ','s')
conjuntoTarget = load(archivoTarget)

%valorAClasificar = input('Ingrese el vector de entrada que quiere clasificar: ')
archivoVectoresClasificar = input('Ingrese el archivo con los vectores a clasificar: ', 's')
vectoresClasificar = load(archivoVectoresClasificar)


%Leer condiciones de finalización
epochMax=input('Ingrese el número máximo de épocas: ')
Eepoch=input('Ingrese el error mínimo: ')
alfa=input('Ingrese el factor de aprendizaje alfa: ')

%Archivo de salida
salida_ar=append('salida-',archivoDataSet)

%Se obtienen el número de valores de entrada y target
[numVectores,  numValores] = size(conjuntoDeEntrenamiento)

%Se determina la matriz de pesos 1xR
w=rand(1, numValores)

%Se determina el valor del bias 1x1
bias=1

%acumulador de w
cont = 1

%Archivo de log
salida=fopen(salida_ar,'w')
%Archivo de errores
errores=fopen('errores_rna.txt','w')
%archivo de los valores de b
ev_bias=fopen('evolucion_bias.txt','w')
%archivo de los valores de w
ev_pesos=fopen('evolucion_pesos.txt','w')

fprintf(ev_bias,' %f', bias);
fprintf(ev_pesos,' %f', w);
fprintf(ev_pesos,';\n');

%Se itera hasta el número máximo de épocas
fprintf("\n>>> Iniciando...<<<")
terminflag = 0
for i=0:epochMax
    fprintf(salida, '\n Época %d \n', i);
    %error acumulado
    eepoch = 0
    %Se itera sobre el conjunto de datos de aprendizaje
    for x=1:numVectores
        %Se obtiene el valor de n = wp+b
        n = (w*transpose(conjuntoDeEntrenamiento(x,1: numValores)))+bias
        a = purelin(n) 
        %Se obtiene e = t-a
        e = conjuntoTarget(x)-a
        %Se acumula el error
        eepoch = eepoch+e
        %Si el error es diferente de 0 se utiliza la regla de aprendizaje
        if e ~= 0
            w = w+((2*alfa)*e*conjuntoDeEntrenamiento(x,1: numValores))
            bias = bias+(2*alfa*e)
            fprintf(ev_bias,' %f', bias);
            fprintf(ev_pesos,' %f', w);
            fprintf(ev_pesos,';\n');
        end
        fprintf(salida,'\nW(%d) = ',cont);
        for k=1: numValores
            fprintf(salida,' %f', w(k))
        end
        fprintf(salida,'\nBias(%d) = %f',cont,bias);
        fprintf(salida,'\nError(%d) = %f',cont,e);
        cont = cont+1
    end
    %Ya que se terminó una época se calcula el error acumulado 
    %1/N(sumaErrores)   
    eepoch = (abs((eepoch)))/(numVectores)
    fprintf(salida,'\nError acumulado(%d) = %f\n',i,eepoch);
    fprintf(errores,' %f',eepoch);
 
    %Se verifican los criterios de finalización
    if eepoch == 0
        terminflag = 2
        break
    elseif eepoch <= Eepoch
        terminflag = 1
        break
    end
    
end
fclose(ev_pesos);
fclose(ev_bias);
fclose(errores);

%Se escribe el criterio de finalización de la red
if terminflag == 0
    fprintf(salida,'\n<<<Terminó por número máximo de iteraciones (epochMax)>>>\n');    
elseif terminflag == 1
    fprintf(salida,'\n<<<Terminó por criterio de finalización eepoch<=Eepoch>>>\n') ;   
elseif terminflag == 2
    fprintf(salida,'\n<<<Terminó por criterio de finalización eepoch=0>>>\n')  ;  
end 

%Una vez que se terminó de entrenar la red se almacenan los valores de los
%pesos y bias 
fprintf(salida,'\nValores finales\nW = ');
for i=1: numValores
    fprintf(salida,' %f', w(i));
end
fprintf(salida,'\nBias = %f\n', bias);

%Se grafica el error
errores_grafica = load('errores_rna.txt')
ev_bias_grafica = load('evolucion_bias.txt')
ev_pesos_grafica = load('evolucion_pesos.txt')
subplot(3,1,1)
plot(errores_grafica)
title('Convergencia del error')
xlabel('Épocas')
ylabel('Error acumulado')
 
subplot(3,1,2)
plot(ev_pesos_grafica)
title('Gráfica de W')

subplot(3,1,3)
plot(ev_bias_grafica)
title('Gráfica de b')

fprintf(salida,'\nPrueba de resultados\n');
%Se prueban los resultados obtenidos
for i=1:numVectores
    n = (w*transpose(conjuntoDeEntrenamiento(i,1: numValores)))+bias
    a = purelin(n)
    fprintf(salida,'\nDato(%d) = %f\n',i, a);
end


[numVectores,  numValores] = size(vectoresClasificar)
fprintf(salida,'\nClasificación de vectores de entrada\n');
for i=1:numVectores
    n = (w*transpose(vectoresClasificar(i,1: numValores)))+bias
    a = purelin(n)
    fprintf(salida,'\nVector(%d) = %f\n',i, a);
end




fclose(salida);
