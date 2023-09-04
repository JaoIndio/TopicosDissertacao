# Elementos Presentes na Espectroscopia
----------------------------

![Alt text](./diagrama.svg)

<!--ts-->
  -[Emissao](#Emissao)<br />
  -[Manipulacao](#Manipulacao)<br />
  -[Reacao](#Reacao)<br />
  -[Deteccao](#Deteccao)<br />
  -[Processamento](#Processamento)<br />
<!--te-->

## Emissao
------------------
  Diz respeito a fonte geradora de luz.

    -Qual o intervalo de comprimento de onda é o mais adequado?
      *Leds?
      *Tungstênio?
      *Halogênio?
    
    -Como dimencionar os colimadores?
    -Como avaliar a necessidade de se usar um colimador?
    -Quais são as necessidades eletrônicas para se acionar a fonte de luz?
    -`Existe algum parametro óptico que possa ser usado como referência de controle
     para alterar o comportamento da fonte de luz?`

## Manipulacao
------------------
  Trata-se dos processos que fazem uso de difração, foco, colimação, interferência
  da luz proveniente da fonte.
    
    -Como dimencionar os angulos dos espelhos?
    -Como dimencionar as distâncias?
    -Quais propriedades ópticas causam rúido, interferência, perdas?
    -`Vale a pena explorar uma relação de servo motores para aproximar, afastar,
      ou alterar os angulos verticais e horizontais dos espelhos?`
    -Qual fenomeno óptico deve ser mensurado (reflectancia, absorção, difração, turbidez)? 

## Reacao
------------------
<!--ts-->
  -[Referencias_Reacao](#Referencias_Reacao)<br />
<!--te-->
  Caracteriza-se por descrever [1] o perfil [2] comportamental [3] do fósforo (picos, intervalos de 
  comprimento de onda usados para quantificação).  
    
    -Qual é o perfil espectrométrico do fósforo?
    -Quais são seus picos?
    -Como quantificar ele?

### Referencias_Reacao
------------------
<a name="sec1"></a> 
####[1]: Nome1
  [1]:#sec1
<a name="section2"></a>
  [2]: #section2 [ 2 ]: Nome2
<a name="section3"></a>
  [2]: #section3 [ 3 ]: Nome3

## Deteccao
------------------
  Elemento(s) utilizados para converter uma grandeza óptica (reflectância, difração, 
  tranmissão, etc) em sinal elétrico.

    -Fotodio? Array de fotodiodos? Outros materiais?
  
  **Depois de responder sobre Emissão, Manipulação e Reação será possivel avaliar**

## Processamento
------------------
  Eletrônica, aloritimos e recuros computacionais (comunicação, RTOS, uC, etc) necessários
  para manipular os sinais elétricos.

    -Quais circuitos serão necessários?
    -Quais processamentos de dados serão necessários (filtros SG, PCA, PLSR)?

  **Designs anteriores e os fundamentos de espectroscopia são o caminho para responder essas perguntas**
