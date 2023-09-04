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
  Caracteriza-se por descrever [1] o perfil [2] comportamental [3] do [4] fósforo (picos, intervalos de 
  comprimento de onda usados para quantificação).  
    
    -Qual é o perfil espectrométrico do fósforo?
    -Quais são seus picos?
    -Como quantificar ele?

  [6] a b

### Referencias_Reacao
------------------
<a name="sec1"></a> 
##### [1]: Iron oxides and organic matter on soil phosphorus availability <br />
  [1]:#sec1
<a name="sec2"></a>
##### [2]: ATR–FTIR Spectroscopic Investigation on Phosphate Adsorption Mechanisms at the Ferrihydrite–Water Interface <br />
  [2]: #sec2 
<a name="sec3"></a>
##### [3]: Phosphorus speciation and distribution in a variable-charge Oxisol under no-till amended with lime and/or phosphogypsum for 18 years<br />
  [3]: #sec3 
<a name="sec4"></a> 
##### [4]: Phosphorus speciation by P-XANES in an Oxisol under long-term no-till cultivation<br />
  [4]:#sec4
<a name="sec5"></a>
##### [5]: Phosphorus speciation in soils with low to high degree of saturation due to swine slurry application<br />
  [5]: #sec5 
<a name="sec6"></a>
##### [6]: On the tropical soils; The influence of organic matter (OM) on phosphate bioavailability<br />
  [6]: #sec6

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
