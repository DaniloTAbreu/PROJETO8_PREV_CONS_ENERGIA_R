# 1 - Definindo o problema de negócio

#Criação de modelos preditivos para
#a previsão de consumo de energia de eletrodomésticos


# 2 - Decisões

#2.1 Informações iniciais

#Nesse projeto de aprendizado de máquina você deve realizar a filtragem de
#dados para remover parâmetros não-preditivos e selecionar os melhores recursos
#(melhores features) para previsão.
#Recomendamos usar RandomForest para a seleção de atributos e SVM, Regressão
#Logística Multilinear ou Gradient Boosting para o modelo preditivo.
#Recomendamos ainda o uso da linguagem R.
#Os dados de energia foram registrados a cada 10 minutos com medidores de
#energia de barramento m.

#2.2 Decisões
#O problema de negócio já informa que é requerido um modelo de Machine Learning. 
#Pelas informações fornecidas, deduzi que, no dataset, a coluna "Appliances" 
#é a variável que queremos prever. 
#Desta forma, iremos utilizar aprendizagem supervisionada.
# Resolvi seguir a recomendação de utilizar a linguagem R.
# Resolvi seguir a recomendação de utilizar RandomForest para a seleção de atributos.
# Resolvi seguir a recomendação de utilizar SVM, Regressão
#Logística Multilinear ou Gradient Boosting para o modelo preditivo
#Falta definir pré processamento.
#O trabalho que será entregue é o melhor modelo encontrado para solução 
#deste problema.

# 3 - Definindo o diretório de trabalho
setwd("C:/Users/Chilov/FCD/PROJETOS/PROJETO8")
getwd()


# 4 - Instalando os pacotes
#install.packages("ggplot2")
#install.packages("corrplot")
#install.packages("dplyr")
#install.packages("e1071")
#install.packages("randomForest")
#install.packages("xgboost")
#install.packages("car")
#install.packages("ModelMetrics")
#install.packages("kable")

# 5 - Carregando os pacotes
library(ggplot2)#permite gráficos
library(corrplot)#permite gráfico de correlação
library(dplyr)#pré processamento de dados
library(e1071) #para modelo SVM
library(randomForest)#para modelo Random Forest
library(xgboost) #para modelo XGBOOST
library(car) #para VIF
library(ModelMetrics)#permite calcular MAE e RMSE
library(knitr)#permite a função kable

# 6 - Dicionário de dados
#[1] "date" - data                    
#[2] "Appliances" - medição de energia dos eletrodomésticos 
#[3] "lights" - uso de energia utilizada por luminárias 
#[4] "T1" - 1ª medição de temperatura                              
#[5] "RH_1" - 1ª medição de umidade relativa       
#[6] "T2" - 2ª medição de temperatura                
#[7] "RH_2" - 2ª medição de umidade relativa                  
#[8] "T3" - 3ª medição de temperatura                
#[9] "RH_3" - 3ª medição de umidade relativa 
#[10] "T4" - 4ª medição de temperatura                
#[11] "RH_4" - 4ª medição de umidade relativa 
#[12] "T5" - 5ª medição de temperatura                 
#[13] "RH_5" - 5ª medição de umidade relativa 
#[14] "T6" - 6ª medição de temperatura                
#[15] "RH_6" - 6ª medição de umidade relativa 
#[16] "T7" - 7ª medição de temperatura                
#[17] "RH_7" - 7ª medição de umidade relativa 
#[18] "T8" - 8ª medição de temperatura                 
#[19] "RH_8" - 8ª medição de umidade relativa 
#[20] "T9" - 9ª medição de temperatura                
#[21] "RH_9" - 9ª medição de umidade relativa 
#[22] "T_out" - previsão do tempo de uma estação de um aeroporto
#[23] "Press_mm_hg" - pressão em mmHg
#[24] "RH_out" - previsão da umidade relativa de uma estação de um aeroporto
#[25] "Windspeed" - velocidade do vento
#[26] "visibility" - visibilidade
#[27] "Tdewpoint" - temperatura para a qual a umidade relativa 
#de uma quantidade de ar atinge 100%
#[28] "rv1" - variável aleatória 1
#[29] "rv2" - variável aleatória 2
#[30] "NSM" - ?
#[31] "WeekStatus" - status da semana
#[32] "Day_of_week" - dia da semana
#variável target está coluna 2 - Appliances


# 7 - Carregando os datasets de treino e de teste convertendo em dataframe
treino_original <- read.csv("projeto8-training.csv")
teste_original <- read.csv("projeto8-testing.csv")
#para análise exploratória, precisamos de todos os dados. Vamos juntá-los
df_original <- rbind(treino_original, teste_original)
#visualiza primeiras linhas
head(treino_original)
head(teste_original)
head(df_original)
#visualiza resumo estatístico
summary(df_original)
length(unique(df_original$lights))
#percebe-se que a coluna ligths neste caso não pode ser considerada numérica
#e sim categórica
treino_original$lights <- as.factor(treino_original$lights) 
teste_original$lights <- as.factor(teste_original$lights) 
df_original$lights <- as.factor(df_original$lights)
#outra forma de visualizar nomes das colunas e valores
str(df_original)
#nomes das colunas
names(df_original)
#dimensões do dataframe
dim(df_original)# 19735 x 32
#visualiza dados em forma de tabela
View(df_original)
#algumas variáveis podem ser excluídas porque não influirão na previsão de 
#consumo de energia
#como não descobri o que é a variável NSM vou excluí-la também
df_1 <- subset(df_original, select = -c(date,WeekStatus,Day_of_week,NSM))
treino_1 <- subset(treino_original, select = -c(date,WeekStatus,Day_of_week,NSM))
teste_1 <- subset(teste_original, select = -c(date,WeekStatus,Day_of_week,NSM))
View(df_1)
str(df_1)
dim(df_1)#19735 28
dim(treino_1)# 14803    28
dim(teste_1)#4932   28
str(df_1)
#todas as variáveis são numéricas e isso está correto
View(treino_1)
#visualiza variável target
head(df_1["Appliances"])
#removendo dados que não precisamos, liberando memória
rm(df_original, treino_original, teste_original)


# 8 - EDA Análise exploratória de dados

# 8.1 verificar valores duplicados
sum(duplicated(df_1))
#não há valores duplicados

# 8.2 verificar valores missing e NA
sapply(df_1, function(y) sum(is.na(y)))
#não há dados missing
summary(df_1)
#como conseguiu realizar os cálculos não temos valor NA


## 8.3 Modelo de Random Forest para buscar as melhores variáveis
#como são muitas variáveis decidi primeiro selecionar as variáveis e depois
#explorarmos elas com mais profundidade
## Padronização - Em Random Forest não é necessário padronizar os dados
set.seed(123)
rf.fit <- randomForest(Appliances ~ ., data=df_1, ntree=50,
                       keep.forest=FALSE, importance=TRUE)

### Visualizando a importância das variáveis

# Obtendo a importância das variáveis do modelo de treino
ImpData <- as.data.frame(importance(rf.fit))
ImpData$Var.Names <- row.names(ImpData)

ggplot(ImpData, aes(x=Var.Names, y=`%IncMSE`)) +
  geom_segment( aes(x=Var.Names, xend=Var.Names, y=0, yend=`%IncMSE`), color="skyblue") +
  geom_point(aes(size = IncNodePurity), color="blue", alpha=0.6) +
  theme_light() +
  coord_flip() +
  theme(
    legend.position="bottom",
    panel.grid.major.y = element_blank(),
    panel.border = element_blank(),
    axis.ticks.y = element_blank()
  )

#A regressão florestal aleatória em R fornece duas saídas: 
#  diminuição no erro quadrático médio (MSE) e pureza do nó. 
#O erro de previsão descrito como MSE é baseado na permutação de 
#seções prontas dos dados por árvore individual e preditor, 
#e os erros são então calculados. No contexto de regressão, 
#a pureza do nó é a diminuição total na soma residual dos quadrados 
#ao dividir em uma variável calculada a média de todas as árvores 
#(ou seja, quão bem um preditor diminui a variância). 
#O MSE é uma medida mais confiável de importância variável. 
#Se as duas métricas de importância mostrarem resultados diferentes, ouça o MSE.
#por este método as 7 variáveis importantes são:
#lights, T8, T5, RH_5, RH_8, RH_9 e Press_mm_hg 
#(SEM DADOS PADRONIZADOS)


#8.4 Explorando as variáveis numéricas de interesse


## 8.4.1 Análise univariada

#Explorando variável categórica
#lights
#Tabela de frequência e proporções desta tabela
table(df_1$lights)
model_table <- table(df_1$lights)
round(model_table, digits = 2)
sort(prop.table(model_table), decreasing=TRUE)
#criando gráfico de barras
barplot(table(df_1$lights), main = "Gráfico de Barras", xlab = "Categorias", ylab = "Contagem", col = "skyblue", ylim = c(0, max(table(df_1$lights)) + 1))
#A maior parte dos valores é zero (sem utilizar energia) e depois 10 e 20.

# Explorando variáveis numéricas

#T8
boxplot(df_1$T8, main = "Boxplot para 8ª medida de temperatura")
#aparentemente possui valores outliers inferiores
hist(df_1$T8,main = "Histograma para 8ª medida de temperatura")
#parece distribuição normal
round (median(df_1$T8,2)) #mediana 22
round (mean(df_1$T8),2) #média 22
round (sd (df_1$T8), 2) #desvio padrão 2
round (var (df_1$T8), 2) #variância 3,8
round(range(df_1$T8), 2)#vai de 16 a 27
diff(range(df_1$T8))# variação total de 11
iqr <- round(IQR(df_1$T8), 2)
iqr #diferença interquartil 10
Q3 <- round(quantile(df_1$T8, probs=0.75), 2)
Q3 # terceiro quartil 23
Q1 <- round(quantile(df_1$T8, probs=0.25), 2)
Q1 # primeiro quartil 21
outliersup_T8 <- Q3+(1.5*iqr) 
outliersup_T8# valores acima de 27 são outliers
outlierinf_T8 <- Q1-(1.5*iqr) 
outlierinf_T8# valores abaixo de 17 são outliers
length(which(df_1$T8 >= outliersup_T8))#não há valor outlier superior
length(which(df_1$T8 <= outlierinf_T8))#temos 79 valores outliers inferiores
#decisão: apesar de serem valores medidos, iremos remover os outliers porque influenciam
#nos modelos de ML ques testaremos (SVM, regressão logística e XGBOOST)

#T5
boxplot(df_1$T5, main = "Boxplot para 5ª medida de temperatura")
#aparentemente possui valores outlierssuperiores
hist(df_1$T5,main = "Histograma para 5ª medida de temperatura")
#não parece distribuição normal
round (median(df_1$T5,2)) #mediana 19
round (mean(df_1$T5),2) #média 20
round (sd (df_1$T5), 2) #desvio padrão 1,8
round (var (df_1$T5), 2) #variância 3,4
round(range(df_1$T5), 2)#vai de 15 a 26
diff(range(df_1$T5))# variação total de 10
iqr <- round(IQR(df_1$T5), 2)
iqr #diferença interquartil 10
Q3 <- round(quantile(df_1$T5, probs=0.75), 2)
Q3 # terceiro quartil 21
Q1 <- round(quantile(df_1$T5, probs=0.25), 2)
Q1 # primeiro quartil 18
outliersup_T5 <- Q3+(1.5*iqr) 
outliersup_T5# valores acima de 24 são outliers
outlierinf_T5 <- Q1-(1.5*iqr) 
outlierinf_T5# valores abaixo de 15 são outliers
length(which(df_1$T5 >= outliersup_T5))#temos 179 valores outliers superiores
length(which(df_1$T5 <= outlierinf_T5))#não há valor outlier inferior
#decisão: apesar de serem valores medidos, iremos remover os outliers porque influenciam
#nos modelos de ML ques testaremos (SVM, regressão logística e XGBOOST)

# RH_5
boxplot(df_1$RH_5, main = "Boxplot para 5ª medição de umidade relativa")
#temos valores outliers inferiores e superiores
hist(df_1$RH_5,main = "Histograma para 5ª medição de umidade relativa")
#parece distribuição normal
round (median(df_1$RH_5),2) #mediana 49
round (mean(df_1$RH_5),2) #média 51
round (sd (df_1$RH_5), 2) #desvio padrão 9  
round (var (df_1$RH_5), 2) #variância 81
round(range(df_1$RH_5), 2)#vai de 30 a 96 
diff(range(df_1$RH_5))# variação total de 67
iqr <- round(IQR(df_1$RH_5), 2)
iqr #diferença interquartil 
Q3 <- round(quantile(df_1$RH_5, probs=0.75), 2)
Q3 # terceiro quartil 54
Q1 <- round(quantile(df_1$RH_5, probs=0.25), 2)
Q1 # primeiro quartil 45
outliersup_RH_5 <- Q3+(1.5*iqr) 
outliersup_RH_5# valores acima de 66 são outliers
outlierinf_RH_5 <- Q1-(1.5*iqr) 
outlierinf_RH_5# valores abaixo de 33 são outliers
length(which(df_1$RH_5 >= outliersup_RH_5))#temos 1314 valores outliers superiores
length(which(df_1$RH_5 <= outlierinf_RH_5))#temos 16 valores outliers inferiores
#decisão: apesar de serem valores medidos, iremos remover os outliers porque influenciam
#nos modelos de ML ques testaremos (SVM, regressão logística e XGBOOST)

# RH_8
boxplot(df_1$RH_8, main = "Boxplot para 8ª medição de umidade relativa")
#temos valores outliers superiores
hist(df_1$RH_8,main = "Histograma para 8ª medição de umidade relativa")
#aparente distribuição normal
round (median(df_1$RH_8),2) #mediana 42
round (mean(df_1$RH_8),2) #média 43
round (sd (df_1$RH_8), 2) #desvio padrão 5,2
round (var (df_1$RH_8), 2) #variância 27
round(range(df_1$RH_8), 2)#vai de 30 a 59
diff(range(df_1$RH_8))# variação total de 29
iqr <- round(IQR(df_1$RH_8), 2)
iqr #diferença interquartil 
Q3 <- round(quantile(df_1$RH_8, probs=0.75), 2)
Q3 # terceiro quartil 47
Q1 <- round(quantile(df_1$RH_8, probs=0.25), 2)
Q1 # primeiro quartil 39
outliersup_RH_8 <- Q3+(1.5*iqr) 
outliersup_RH_8# valores acima de 58 são outliers
outlierinf_RH_8 <- Q1-(1.5*iqr) 
outlierinf_RH_8# valores abaixo de 28 são outliers
length(which(df_1$RH_8 >= outliersup_RH_8))#apenas 17 valores são outliers superiores
length(which(df_1$RH_8 <= outlierinf_RH_8))#não há valores outliers inferiores
#decisão: apesar de serem valores medidos, iremos remover os outliers porque influenciam
#nos modelos de ML ques testaremos (SVM, regressão logística e XGBOOST)

# RH_9
boxplot(df_1$RH_9, main = "Boxplot para 9ª medição de umidade relativa")
#temos valores outliers inferiores e superiores
hist(df_1$RH_9,main = "Histograma para 9ª medição de umidade relativa")
#aparente distribuição normal
round (median(df_1$RH_9),2) #mediana 41
round (mean(df_1$RH_9),2) #média 42
round (sd (df_1$RH_9), 2) #desvio padrão 4,2  
round (var (df_1$RH_9), 2) #variância 17
round(range(df_1$RH_9), 2)#vai de 29 a 53
diff(range(df_1$RH_9))# variação total de 24
iqr <- round(IQR(df_1$RH_9), 2)
iqr #diferença interquartil 
Q3 <- round(quantile(df_1$RH_9, probs=0.75), 2)
Q3 # terceiro quartil 44
Q1 <- round(quantile(df_1$RH_9, probs=0.25), 2)
Q1 # primeiro quartil 38
outliersup_RH_9 <- Q3+(1.5*iqr) 
outliersup_RH_9# valores acima de 53 são outliers
outlierinf_RH_9 <- Q1-(1.5*iqr) 
outlierinf_RH_9# valores abaixo de 30 são outliers
length(which(df_1$RH_9 >= outliersup_RH_9))#apenas 4 valores são outliers superiores
length(which(df_1$RH_9 <= outlierinf_RH_9))#apenas 17 valores são outliers inferiores
#decisão: apesar de serem valores medidos, iremos remover os outliers porque influenciam
#nos modelos de ML ques testaremos (SVM, regressão logística e XGBOOST)

#Press_mm_hg
boxplot(df_1$Press_mm_hg, main = "Boxplot para pressão (mmHg)")
#aparentemente possui valores outliers inferiores
hist(df_1$Press_mm_hg,main = "Histograma para pressão (mmHg)")
#parece distribuição normal
round (median(df_1$Press_mm_hg,2)) #mediana 756
round (mean(df_1$Press_mm_hg),2) #média 756
round (sd (df_1$Press_mm_hg), 2) #desvio padrão 7,4
round (var (df_1$Press_mm_hg), 2) #variância 55
round(range(df_1$Press_mm_hg), 2)#vai de 729 a 772
diff(range(df_1$Press_mm_hg))# variação total de 43
iqr <- round(IQR(df_1$Press_mm_hg), 2)
iqr #diferença interquartil 
Q3 <- round(quantile(df_1$Press_mm_hg, probs=0.75), 2)
Q3 # terceiro quartil 761
Q1 <- round(quantile(df_1$Press_mm_hg, probs=0.25), 2)
Q1 # primeiro quartil 751
outliersup_Press_mm_hg <- Q3+(1.5*iqr) 
outliersup_Press_mm_hg# valores acima de 776 são outliers
outlierinf_Press_mm_hg <- Q1-(1.5*iqr) 
outlierinf_Press_mm_hg# valores abaixo de 736 são outliers
#tenho outliers superiores
length(which(df_1$Press_mm_hg >= outliersup_Press_mm_hg))#não há valores outliers superiores
length(which(df_1$Press_mm_hg <= outlierinf_Press_mm_hg))#apenas 17 valores são outliers inferiores
#decisão: apesar de serem valores medidos, iremos remover os outliers porque influenciam
#nos modelos de ML ques testaremos (SVM, regressão logística e XGBOOST)


## 8.4.2 ANÁLISE BIVARIADA
# Scatterplot 8ª medição de Umidade relativa x  pressão (mmHg)
plot(x = df_1$RH_8 , y = df_1$Press_mm_hg,
     main = "Scatterplot - 8ª medição de Umidade relativa x pressão (mmHg)",
     xlab = "Umidade relativa",
     ylab = "Pressão")
#sem correlação entre RH_8 e Press_mm_hg


# Barplot Energia utilizada por luminárias  x 5ª medição de Umidade relativa
barplot(tapply(df_1$RH_5, df_1$lights, mean), beside = TRUE, main = "Gráfico de Barras Agrupado - Energia utilizada por luminárias x 5ª medição de Umidade relativa", xlab = "Energia utilizada por luminárias", ylab = "Média dos Valores - Umidade relativa", col = rainbow(length(unique(df_1$lights))))
legend("topright", legend = unique(df_1$lights), fill = rainbow(length(unique(df_1$lights))))
#a média de umidade relativa se mantém mesmo alterando a energia gasta 
#pelas luminárias


## 8.4.3 EXCLUSÃO DOS VALORES OUTLIERS
#T8inf, T5sup, RH_5infsup,RH_8sup,RH_9insup e Press_mm_hginf
#T8inf
df_2 <- df_1 [which(df_1$T8 >= outlierinf_T8), ]
treino_2 <- treino_1 [which(treino_1$T8 >= outlierinf_T8), ]
teste_2 <- treino_1 [which(teste_1$T8 >= outlierinf_T8), ]

##T5sup
df_3 <- df_2 [which(df_2$T5 <= outliersup_T5), ]
treino_3 <- treino_2 [which(treino_2$T5 <= outliersup_T5), ]
teste_3 <- teste_2 [which(teste_2$T5 <= outliersup_T5), ]

#RH_5infsup
df_4 <- df_3 [which(df_3$RH_5 >= outlierinf_RH_5), ]
treino_4 <- treino_3 [which(treino_3$RH_5 >= outlierinf_RH_5), ]
teste_4 <- treino_3 [which(teste_3$RH_5 >= outlierinf_RH_5), ]
df_4 <- df_4 [which(df_4$RH_5 <= outliersup_RH_5), ]
treino_4 <- treino_4 [which(treino_4$RH_5 <= outliersup_RH_5), ]
teste_4 <- teste_4 [which(teste_4$RH_5 <= outliersup_RH_5), ]

#RH_8sup
df_5 <- df_4 [which(df_4$RH_8 <= outliersup_RH_8), ]
treino_5 <- treino_4 [which(treino_4$RH_8 <= outliersup_RH_8), ]
teste_5 <- teste_4 [which(teste_4$RH_8 <= outliersup_RH_8), ]

#RH_9insup
df_6 <- df_5 [which(df_5$RH_9 >= outlierinf_RH_9), ]
treino_6 <- treino_5 [which(treino_5$RH_9 >= outlierinf_RH_9), ]
teste_6 <- treino_6 [which(teste_5$RH_9 >= outlierinf_RH_9), ]
df_6 <- df_6 [which(df_6$RH_9 <= outliersup_RH_9), ]
treino_6 <- treino_6 [which(treino_6$RH_9 <= outliersup_RH_9), ]
teste_6 <- teste_6 [which(teste_6$RH_9 <= outliersup_RH_9), ]

#Press_mm_hginf
df_7 <- df_6 [which(df_6$Press_mm_hg >= outlierinf_Press_mm_hg), ]
treino_7 <- treino_6 [which(treino_6$Press_mm_hg >= outlierinf_Press_mm_hg), ]
teste_7 <- treino_6 [which(teste_6$Press_mm_hg >= outlierinf_Press_mm_hg), ]

#removendo dados que não precisamos, liberando memória
rm(df_1,df_2, df_3, df_4,df_5,df_6, treino_1,treino_2,treino_3,treino_4,treino_5,
treino_6, teste_1,teste_2,teste_3,teste_4,teste_5, teste_6)


## 8.4.4 CORRELAÇÃO 
#a correlação avalia somente variáveis numéricas
#vamos ver a correlação das variáveis mais importantes segundo Random Forest
#tirar a variável categorica lights
df_7_num <- subset(df_7, select = c(Appliances, T8, T5, RH_5, RH_8, RH_9, Press_mm_hg))
#correlação e tira da notação científica
L <- cor(df_7_num)
options(scipen = 100, digits = 2)#ajusta para duas casas decimais 
View(L)
#criando mapa de correlação com números coloridos
corrplot(L, method = 'number') 
#Insights:
#Appliance tem baixa correlação negativa com RH_8
#entre as variáveis preditoras, T5 e T8 têm alta correlação (ruim para modelo ML)
#entre as variáveis preditoras, RH_5, RH_8 e RH_9 têm alta correlação (ruim para modelo ML)
#decisão: excluir T5, RH_5 e RH_9 
#as variáveis que sobraram tem uma maior correlação com a variável alvo Appliances


# 9 - Modelo Machine Learning

#Após o final do item 8.4.3, seguiremos com as seguintes variáveis preditoras:
#lights, T8, RH_8, Press_mm_hg 
# Em Random Forest não é necessário padronizar os dados

#9.1 Modelos com SVM
#modelo_v1
#vamos tratar somente com as variáveis importantes segundo o Random Forest e depois de
#tirar as variáveis preditoras com alta correlação entre si:
df_final <- subset(df_7, select = c(lights, T8, RH_8, Press_mm_hg , Appliances))
treino_final <- subset(treino_7, select = c(lights, T8, RH_8, Press_mm_hg , Appliances))
teste_final <- subset(teste_7, select = c(lights, T8, RH_8, Press_mm_hg , Appliances))

#removendo dados que não precisamos, liberando memória
rm(treino_7,teste_7, df_7)
#verificando tamanho, tipo e visualizando os dados
dim(treino_final)#13488 x 5
dim(teste_final)#4264 x 5
View(teste_final)
typeof(treino_final) #list

#na própria fórmula, escolhemos fazer a padronização das variáveis
set.seed(406)
modelo_v1 <- svm(formula = Appliances ~ ., 
                 data = treino_final, type = 'nu-regression', 
                kernel = 'polynomial', na.action = na.omit, scale = TRUE)

summary(modelo_v1)

#Métricas
#Fazendo previsões e verificando acurácia com o modelo
#MAE - Mean Absolute Error - quanto menor, melhor
#RMSE - Root Mean Squared Error - quanto menor, melhor
#R2 - coeficiente de determinação - quanto maior, melhor (cuidado com overfitting)
scores <- data.frame(observado = teste_final$Appliances,
                     prediction = predict(modelo_v1, newdata = teste_final))
scores$diferenca <- scores$observado - scores$prediction
scores$diferenca_abs <- abs(scores$observado - scores$prediction)
print(scores)
#Cálculo das métricas
MAE_v1 <- mae(scores$observado,scores$prediction)
round(MAE_v1, 2)#50
RMSE_v1<-rmse(scores$observado,scores$prediction)
round(RMSE_v1, 2)#110
#coeficiente de determinação
R2_v1 <- cor(scores$observado,scores$prediction)^2
round(R2_v1, 2)#0.16 #muito ruim!


#modelo_v2
#novo modelo mudando o type
set.seed(252)
modelo_v2 <- svm(formula = Appliances ~ ., 
                 data = treino_final, type = 'eps-regression', 
                 kernel = 'polynomial', na.action = na.omit, scale = TRUE)

summary(modelo_v2)

# Fazendo previsões e verificando acurácia com o modelo
scores <- data.frame(observado = teste_final$Appliances,
                     prediction = predict(modelo_v2, newdata = teste_final))
scores$diferenca <- scores$observado - scores$prediction
scores$diferenca_abs <- abs(scores$observado - scores$prediction)
print(scores)
#Cálculo das métricas
MAE_v2 <- mae(scores$observado,scores$prediction)
round(MAE_v2, 2)#49
RMSE <- sqrt(mean((scores$observado - scores$prediction)^2))
round(RMSE, 2)#111
RMSE_v2<-rmse(scores$observado,scores$prediction)
RMSE_v2
#coeficiente de determinação
R2_v2 <- cor(scores$observado,scores$prediction)^2
round(R2_v2, 2)#0.16 muito ruim!


#modelo_v3
#novo modelo mudando o kernel
set.seed(562)
modelo_v3 <- svm(formula = Appliances ~ ., 
                 data = treino_final, type = 'nu-regression', 
                 kernel = 'linear', na.action = na.omit, scale = TRUE)

summary(modelo_v3)

#Fazendo previsões e verificando acurácia com o modelo
scores <- data.frame(observado = teste_final$Appliances,
                     prediction = predict(modelo_v3, newdata = teste_final))
scores$diferenca <- scores$observado - scores$prediction
scores$diferenca_abs <- abs(scores$observado - scores$prediction)
print(scores)
#Cálculo das métricas
MAE_v3 <- mae(scores$observado,scores$prediction)
round(MAE_v3, 2)#112
RMSE <- sqrt(mean((scores$observado - scores$prediction)^2))
round(RMSE, 2)#
RMSE_v3<-rmse(scores$observado,scores$prediction)
RMSE_v3
#coeficiente de determinação
R2_v3 <- cor(scores$observado,scores$prediction)^2
round(R2_v3, 2)#0.13 muito ruim!


#modelo_v4
#novo modelo mudando o kernel
set.seed(101)
modelo_v4 <- svm(formula = Appliances ~ ., 
                 data = treino_final, type = 'nu-regression', 
                 kernel = 'sigmoid', na.action = na.omit, scale = TRUE)

summary(modelo_v4)

#Fazendo previsões e verificando acurácia com o modelo
scores <- data.frame(observado = teste_final$Appliances,
                     prediction = predict(modelo_v4, newdata = teste_final))
scores$diferenca <- scores$observado - scores$prediction
scores$diferenca_abs <- abs(scores$observado - scores$prediction)
print(scores)
#Cálculo das métricas
MAE_v4 <- mae(scores$observado,scores$prediction)
round(MAE_v4, 2)# 2878
RMSE <- sqrt(mean((scores$observado - scores$prediction)^2))
round(RMSE, 2)# 4606
RMSE_v4<-rmse(scores$observado,scores$prediction)
RMSE_v4
#coeficiente de determinação
R2_v4 <- cor(scores$observado,scores$prediction)^2
round(R2_v4, 2)#0.01


#9.2 Modelo de regressão linear múltipla
#modelo_v5
set.seed(450)
modelo_v5 <- glm(Appliances ~ ., treino_final, family = gaussian)
summary(modelo_v5)
#métricas
scores <- data.frame(observado = teste_final$Appliances,
                     prediction = predict(modelo_v5, newdata = teste_final))
scores$diferenca <- scores$observado - scores$prediction
scores$diferenca_abs <- abs(scores$observado - scores$prediction)
print(scores)
#Cálculo das métricas 
MAE_v5 <- mae(scores$observado,scores$prediction)
round(MAE_v5, 2)#58
RMSE <- sqrt(mean((scores$observado - scores$prediction)^2))
round(RMSE, 2)# 105
RMSE_v5<-rmse(scores$observado,scores$prediction)
RMSE_v5
#coeficiente de determinação
R2_v5 <- cor(scores$observado,scores$prediction)^2
round(R2_v5, 2)# 0.14 muito ruim

#verificando multicolinearidade entre variáveis
kable(vif(modelo_v5), align = 'c')
#comprova o que já tinhamos visto no item 8. Não há mais colinearidade
#entre variáveis preditoras


#9.3 Modelo de Boosted Decision Tree Regression 
#modelo_v6
# Construindo o modelo
treino.y <- treino_final$Appliances
teste.y <- teste_final$Appliances
#trocar a variável target de tipo fator para tipo numérico
treino.y <- as.numeric(treino.y) - 1
teste.y <- as.numeric(teste.y) - 1
View(treino.y)
#isolar a variável X (que tem que ser tipo de matriz neste algoritmo)
View(treino_final)
treino.x <- data.matrix(treino_final[,1:4])
teste.x <- data.matrix(teste_final[,1:4])
str(treino.x)
#configurando os parâmetros
set.seed(643)
params <- list(eta = 0.3, max_depth = 6, 
               subsample = 1,
               colsample_bytree=1,
               min_child_weight= 1,
               gamma = 0,
               eval_metric = "rmse", 
               objective = "reg:squarederror",
               booster = "gblinear")

#iniciar o xgboost
modelo_v6 <- xgboost(data = treino.x, label = treino.y, 
                      params = params, 
                      set_seed = 102,
                      nround = 20,
                      verbose = 1)

#avaliando o modelo - métricas
pred <- predict(modelo_v6, newdata = data.matrix(teste.x))
err <- abs(teste.y-pred)
acuracia <- sum(err)/length(err)
print(paste("test-error = ", err))
round(acuracia, 2)#66
MAE_v6 <- mae(teste.y,pred)
round(MAE_v6, 2)#66
RMSE_v6 <- sqrt(mean((teste.y - pred)^2))
round(RMSE_v6, 2)#107
R2_v6 <- cor(teste.y,pred)^2
round(R2_v6, 2)#0,11


#modelo_v7
#vamos alterar o parâmetro booster para tentar melhorar a acurácia
set.seed(338)
params <- list(eta = 0.3, max_depth = 6, 
               subsample = 1,
               colsample_bytree=1,
               min_child_weight= 1,
               gamma = 0,
               eval_metric = "rmse", 
               objective = "reg:squarederror",
               booster = "dart")

#iniciar o xgboost
modelo_v7 <- xgboost(data = treino.x, label = treino.y, 
                      params = params, 
                      set_seed = 102,
                      nround = 20,
                      verbose = 1)

#avaliando o modelo - métricas
pred <- predict(modelo_v7, newdata = data.matrix(teste.x))
err <- abs(teste.y-pred)
acuracia <- sum(err)/length(err)
print(paste("test-error = ", err))
round(acuracia, 2)#42
MAE_v7 <- mae(teste.y,pred)
round(MAE_v7, 2)#42
RMSE_v7 <- sqrt(mean((teste.y - pred)^2))
round(RMSE_v7, 2)#77
R2_v7 <- cor(teste.y,pred)^2
round(R2_v7, 2)#0,57


#modelo_v8
#vamos alterar o parâmetro booster para tentar melhorar a acurácia
params <- list(eta = 0.3, max_depth = 6, 
               subsample = 1,
               colsample_bytree=1,
               min_child_weight= 1,
               gamma = 0,
               eval_metric = "rmse", 
               objective = "reg:squarederror",
               booster = "gbtree")


#iniciar o xgboost
set.seed(879)
modelo_v8 <- xgboost(data = treino.x, label = treino.y, 
                     params = params, 
                     set_seed = 102,
                     nround = 20,
                     verbose = 1)

#avaliando o modelo
pred <- predict(modelo_v8, newdata = data.matrix(teste.x))
err <- abs(teste.y-pred)
acuracia <- sum(err)/length(err)
print(paste("test-error = ", err))
round(acuracia, 2)#42
MAE_v8 <- mae(teste.y,pred)
round(MAE_v8, 2)#42
RMSE_v8 <- sqrt(mean((teste.y - pred)^2))
round(RMSE_v8, 2)#77
R2_v8 <- cor(teste.y,pred)^2
round(R2_v8, 2)#0,57

#resultados iguais ao do modelo_v7
#parâmetro booster'dart' e 'gbtree' apresentaram a mesma acurácia

#RESUMO
#DECISÃO:até o momento, o melhor modelo foi do xgboost
#vamos tentar mudar a acurácia otimizando os hiper parâmetros


# 9.4 Otimizaçãõ de hiperparâmetros de Modelo de Machine Learning
#vamos obter os melhores hiperparâmetros sem utilizar xbg.cv
#dados  de treino e respectivos labels
dados <- treino.x
labels <- as.vector(unlist(treino_final$Appliances))

# Grid de hiperparâmetros a serem testados
param_grid <- list(
  eta = c(0.1, 0.2, 0.3),
  max_depth = c(4, 6, 8),
  subsample = c(0.8, 1),
  colsample_bytree = c(0.8, 1),
  min_child_weight = c(1, 3, 5),
  gamma = c(0, 0.1, 0.2)
)

# Divide os dados em treinamento e validação
set.seed(123) # para garantir a reprodutibilidade
indices <- sample(1:nrow(dados), 0.8 * nrow(dados)) # 80% dos dados para treinamento
dados_treinamento <- dados[indices, ]
labels_treinamento <- labels[indices]
dados_validacao <- dados[-indices, ]
labels_validacao <- labels[-indices]

# Inicializa os melhores hiperparâmetros e a melhor métrica
melhor_metrica <- Inf
melhores_hiperparametros <- NULL

# Percorre a grade de hiperparâmetros
for (eta in param_grid$eta) {
  for (max_depth in param_grid$max_depth) {
    for (subsample in param_grid$subsample) {
      for (colsample_bytree in param_grid$colsample_bytree) {
        for (min_child_weight in param_grid$min_child_weight) {
          for (gamma in param_grid$gamma) {
            # Treina o modelo com os hiperparâmetros atuais
            modelo <- xgboost(data = dados, label = labels,
                              eta = eta, max_depth = max_depth,
                              subsample = subsample, colsample_bytree = colsample_bytree,
                              min_child_weight = min_child_weight, gamma = gamma,
                              eval_metric = "rmse", objective = "reg:squarederror",
                              nrounds = 100, verbose = FALSE)
            
            # Avalia o modelo nos dados de validação
            predicoes <- predict(modelo, newdata = dados_validacao)
            metrica <- sqrt(mean((predicoes - labels_validacao)^2)) # RMSE
            
            # Verifica se a métrica atual é melhor do que a melhor métrica anterior
            if (metrica < melhor_metrica) {
              melhor_metrica <- metrica
              melhores_hiperparametros <- list(eta = eta, max_depth = max_depth,
                                               subsample = subsample,
                                               colsample_bytree = colsample_bytree,
                                               min_child_weight = min_child_weight,
                                               gamma = gamma)
            }
          }
        }
      }
    }
  }
}

# Visualiza os melhores hiperparâmetros encontrados
print(melhores_hiperparametros)
#eta=0.3
#max depth=8
#subsample = 0.8
#colsample_bytree = 1 
#min_child_weight = 1
#gamma = 0.2
#eval_metric = rmse
#objective = "reg:squarederror",
#booster = "gbtree"

#vamos utilizar os parâmetros otimizados
params <- list(eta = 0.3, max_depth = 8, 
               subsample = 0.8,
               colsample_bytree=1,
               min_child_weight= 1,
               gamma = 0.2,
               eval_metric = "rmse", 
               objective = "reg:squarederror",
               booster = "gbtree")

set.seed(879)
modelo_final <- xgboost(data = treino.x, label = treino.y, 
                      params = params, 
                      set_seed = 102,
                      nround = 100,
                      verbose = 1)

#avaliando o modelo
pred <- predict(modelo_final, newdata = data.matrix(teste.x))
err <- abs(teste.y-pred)
acuracia <- sum(err)/length(err)
print(paste("test-error = ", err))
round(acuracia, 2)#18
MAE_final <- mae(teste.y,pred)
round(MAE_final, 2)#18
RMSE_final <- sqrt(mean((teste.y - pred)^2))
round(RMSE_final, 2)#31
R2_final <- cor(teste.y,pred)^2
round(R2_final, 2)#0.93

#CONCLUSÃO: O MELHOR MODELO é o modelo_final 
#que teve os parâmetros XGBOOST otimizados.
