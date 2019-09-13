# cnn

Proiectul reprezinta implementarea unei retele neurale convolutionale pentru recunoasterea cifrelor scrise de mana. 
Este scris in limbajul C++; fisierul sursa este convnet.cpp; foloseste pentru antrenare si testare setul de date MNIST. 
Momentan nu este complet. 

Reteaua neurala este implementata astfel: 
Am folosit o clasa abstracta Layer - reprezinta un tip de layer (nivel) din retea - contine metodele: forward(), backProp() si updateWeights(). Din aceasta clasa am derivat apoi mai multe clase, care corespund diverselor tipuri de layere: de convolutie, de pooling, nivele complet conectate (fully connected layer), ReLU etc. Am separat nivelurile de convolutie si cele complet conectate, considerand nivelul ReLU ca un nivel separat, pentru simplitate, usurinta implementarii si modularitate. Desi se numeste ReLU, nivelul respectiv aplica de fapt functia sigmoid asupra intrarii. 
Reteaua este alcatuita dintr-un vector de astfel de nivele. Fiecare nivel cate doua referinte, una pentru datele de intrare si cealalta pentru datele de iesire din nivel. 
Un neuron este reprezentat printr-o structura continand valoarea neuronului si gradientul erorii delta. Datele sunt organizate fie intr-un vector unidimensional de astfel de structuri (clasa Data1D), fie in tablouri tri-dimensionale (clasa Data3D). 
Pentru antrenarea retelei se aplica la intrarea acesteia imaginea de intrare, apoi se apeleaza succesiv pentru fiecare nivel din retea metoda forward(), apoi se compara iesirea cu valoarea reala si se propaga eroarea inapoi in retea, apelandu-se backProp() pentru fiecare nivel in ordine inversa. forward() modifica valorile neuronilor de iesire al nivelului, iar backProp() modifica gradientul erorii pentru neuronii de iesire. In final se apeleaza updateWeights() pentru toate nivelele (cele care au ponderi) si se actualizeaza valoarea ponderilor in functie de gradientul erorii, folosind metoda Gradient Descent. 

