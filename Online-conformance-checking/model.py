
# TO DOOOOO : think how to estimate a marking for a non-conform transition (wether coming from a conformant state or non-conformant state)
"""
marking estimation <=> 'very not likely jump' => very high cost : 
distance between the current amrking and marking before a non-conformant transition
"""


# movements required to align to a conformant sequence should be telling the relashionship between paths
# if i get to express : mov1 + mov2 = mov3 <=> A model can learn to minimize the number of movements (shortest path) 

#...............................................................................
# Loss = Erreur de Position + coefficient * Nombre de saut des marking conformes
#...............................................................................

#...............................................................................
# Loss de position = 1 - cos(position estimée, position cible)
#...............................................................................


