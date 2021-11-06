#############################################################
# Script de máquinas de aprendizaje
# Con datos de:
#   SUPERVISADO: https://www.kaggle.com/adityadesai13/used-car-dataset-ford-and-mercedes
#   NO SUPERVISADO: https://www.kaggle.com/shilpagopal/us-crime-dataset, código de: https://www.statology.org/principal-components-analysis-in-r/
# Para el curso Taller comercio y desarrollo regional
# (Módulo Aprendizaje de máquinas)
# Educación Continua | Universidad de los Andes
# 20 de octubre 2021
#############################################################

######################################
# AGENDA:
# 1. Problemas de regresión
#     - Regresión lineal
# 2. Problemas de clasificación
#     - Clasificación con regresión logística (es similar con lineal)
#     - K-vecinos más cercanos
#     - Arbol de decición
# 3. Problemas de aprendizaje no supervisado
#     - Análisis de componentes principales (PCA)

# Quedan por fuera: clusterización: k-medias jerárquica
######################################

########################
# Preliminares
########################

# Para poder cargar un archivo excel a R instalamos:
#install.packages('readxl')
#library('readxl')

# Base de datos de precios de autos Audi
setwd("/Users/alorozco22/OneDrive - Universidad de los Andes/Documentos 2021-20/2021 08 30 EdCo Curso Contraloría ML Fayber/repo-comercio/data")

audi <- read.csv(file="audi.csv")
View(audi)

crimen <- read.delim(file="uscrime.txt")
View(crimen)

# FILAS: 47 estados de EEUU
# M - percentage of males aged 14–24 in total state population
# So - indicator variable for a southern state
# Ed - mean years of schooling of the population aged 25 years or over
# Po1 - per capita expenditure on police protection in 1960
# Po2 - per capita expenditure on police protection in 1959
# LF - labour force participation rate of civilian urban males in the age group 14-24
# M.F - number of males per 100 females
# Pop - state population in 1960 in hundred thousand
# NW - percentage of nonwhites in the population
# U1 - unemployment rate of urban males 14–24
# U2 - unemployment rate of urban males 35–39
# wealth - median value of transferable assets or family income
# Ineq - income inequality: percentage of families earning below half the median income
# Prob - probability of imprisonment: ratio of number of commitments to nunumber of offenses
# Time - average time in months served by offenders in state prisons before their first release
# Crime - crime rate: number of offenses per 100,000 population in 1960
colnames(crimen) <- c("hombres", "sur", "educ", "gasto60", "gasto59", "fuerzaLabo", "hombresPor100Mujeres", "Poblacion", "No blancos", "Desempleo14a24", "Desempleo35a39", "riqueza", "desigualdad", "encarcelamiento", "mesesPrision", "crimen")
################################################
# Preprocesamos & dividimos en train(70%) test(30%)
################################################

# PRIMERO: PARA EL PROBLEMA DE CLASIFICACIÓN
# Vamos a predecir si el auto es de tipo diesel o gasolina
table(factor(audi$fuelType))
# Como hay tres tipos de auto, vamos a predecir gasolina vs los demás.
# Primero construimos una columna que indique si el auto es de gasolina
audi$gasolina <- ifelse(audi$fuelType=="Petrol", 1, 0)
table(audi$gasolina)

View(audi)

# SEGUNDO: Como vamos a usar K vecinos más cercanos, de paso:
# Este algoritmo requiere normalizar los datos para lidiar con escalas
# Normalizamos las columnas predictoras (dato - mín)/(máx - min)
audi$engineSizeN <- (audi$engineSize - min(audi$engineSize))/(max(audi$engineSize)- min(audi$engineSize))
audi$priceN <- (audi$price - min(audi$price))/(max(audi$price)- min(audi$price))

# SEPARAMOS EN DATOS DE ENTRENAMIENTO Y PRUEBA
set.seed(222) # Semilla para replicar proceso aleatorio
ind <- sample(2, nrow(audi), replace = TRUE, prob = c(0.7, 0.3))
head(ind)
# Armamos los dataframes con el indicador de grupo 1 o grupo 2
trainingAudi <- audi[ind==1,]
testingAudi <- audi[ind==2,]

################################################
# Entrenamientos, predicciones y evaluaciones rápidas
################################################

# 1. Problemas de regresión

# Veamos qué variables pueden tener potencial para predecir el precio 
install.packages("corrgram")
library("corrgram")
corrgram(audi, lower.panel=panel.shade, upper.panel=panel.cor)
# Todas las variables tienen muy buenas correlaciones!

#     - Regresión lineneal
#       precio  = b0 + b1(millas por galon)+b2(tañamo de motor)+b3(impuesto)+b4(millas)+b5(año)
        modeloLinealCarros <- lm(price~mpg+engineSize+tax+mileage+year, trainingAudi)
        summary(modeloLinealCarros)
        
        
        # Predicciones con datos nuevos!
        prediccionesTest <- predict(modeloLinealCarros, testingAudi)
        head(prediccionesTest)
        
        # Para evaluar vemos valores reales de precio y predicciones
        evalLinealAudi <- cbind(testingAudi$price, prediccionesTest)
        colnames(evalLinealAudi) <- c('Real', 'Predicho')
        evalLinealAudi <- as.data.frame(evalLinealAudi)
        View(evalLinealAudi)
        
        # Promedio de errores
        errorPromedio <- mean((evalLinealAudi$Real - evalLinealAudi$Predicho))
        # Error al cuadrado promedio
        errorPromedio2 <- mean((evalLinealAudi$Real - evalLinealAudi$Predicho)*(evalLinealAudi$Real - evalLinealAudi$Predicho))
        
        
# 2. Problemas de clasificación

  
#     - Clasificación con regresión logística (es similar con lineal)
        modeloLogistico <- glm(gasolina~engineSize+price, data =  trainingAudi, family = "binomial")
        summary(modeloLogistico)
        
        # Predicciones con datos nuevos
        prediccionesTest <- predict(modeloLogistico, newdata = testingAudi, type = "response")
        evalLogitAudi <- cbind(testingAudi$gasolina, prediccionesTest)
        colnames(evalLogitAudi) <- c('Real', 'Predicho')
        evalLogitAudi <- as.data.frame(evalLogitAudi)
        View(evalLogitAudi)
        
        # Para evaluar convertimos a 0 y 1 la predicción:
        evalLogitAudi$GasPredicha <- ifelse(evalLogitAudi$Predicho > 0.5, 1, 0)
        
        # Matriz de confusión
        matriz <- table(evalLogitAudi$GasPredicha, evalLogitAudi$Real)
        # Tasa de error
        error <- 1-sum(diag(matriz))/sum(matriz)
          


#     - K-vecinos más cercanos
        install.packages("class")
        library("class")
        trainingAudiKNN <- as.data.frame(cbind(trainingAudi$engineSizeN, trainingAudi$priceN))
        testingAudiKNN <- as.data.frame(cbind(testingAudi$engineSizeN, testingAudi$priceN))
        colnames(trainingAudiKNN) <- c('engineSizeN', 'priceN')
        colnames(testingAudiKNN) <- c('engineSizeN', 'priceN')
        
        prediccionesKVecinos <- knn(trainingAudiKNN,testingAudiKNN,cl=trainingAudi$gasolina,k=5)
        
        evalKNNAudi <- as.data.frame(cbind(testingAudi$gasolina,prediccionesKVecinos))
        colnames(evalKNNAudi) <- c('Real', 'Predicho')
        
        
        evalKNNAudi$Predicho <- ifelse(evalKNNAudi$Predicho == 1, 0, 2)
        evalKNNAudi$Predicho <- ifelse(evalKNNAudi$Predicho == 2, 1, 0)
        
        # Matriz de confusión
        matriz <- table(evalKNNAudi$Predicho,evalKNNAudi$Real)
        # Tasa de error
        error <- 1-sum(diag(matriz))/sum(matriz)
        
#     - Arbol de decición
        install.packages("tree")
        library("tree")
        trainingAudiArbol <- as.data.frame(cbind(trainingAudi$engineSize, trainingAudi$price, trainingAudi$gasolina))
        testingAudiArbol <- as.data.frame(cbind(testingAudi$engineSize, testingAudi$price, testingAudi$gasolina))
        colnames(trainingAudiArbol) <- c('engineSize', 'price', 'gasolina')
        colnames(testingAudiArbol) <- c('engineSize', 'price', 'gasolina')
        
        arbol <- tree(as.factor(gasolina)~., data=trainingAudiArbol)
        summary(arbol)
        plot(arbol)
        text(arbol, cex=0.5)
        
        prediccionesArbol <- predict(arbol, testingAudiArbol, type="class")
        
        evalArbolAudi <- as.data.frame(testingAudi$gasolina)
        evalArbolAudi$Predicho <- prediccionesArbol
        colnames(evalArbolAudi) <- c('Real', 'Predicho')
        
        # Matriz de confusión
        matriz <- table(evalArbolAudi$Predicho,evalArbolAudi$Real)
        # Tasa de error
        error <- 1-sum(diag(matriz))/sum(matriz)

# 3. Problemas de aprendizaje no supervisado
#     - Análisis de componentes principales (PCA)
        componentes <- prcomp(crimen, scale = TRUE)
        componentes$rotation <- -1*componentes$rotation
        # A qué hace más referencia cada componente
        componentes$rotation
        # PC1: riqueza y desigualdad
        # PC2: población y mesesEnPrisión
        # PC3: desempleo
        # PC4: crimen
        # Cada componente contiene menos varianza que el anterior
        
        # Dónde se ubica cada observación
        componentes$x <- -1*componentes$x
        head(componentes$x)
        
        # Podemos visualizar en los primeros dos componentes
        # Los datos más cercanos presentan patrones similares
        # Hay estados que se diferencian más por algunas características
        # El estado 26 es muy característico por su riqueza
        biplot(componentes, scale = 0, cex=0.5)

        # Qué porcentaje de la varianza de los datos captura cada componente
        componentes$sdev^2 / sum(componentes$sdev^2)
        
        # Entre el primero y el segundo componente explican el
        0.3888874592+0.1830000080
          
          
          
          
          
        