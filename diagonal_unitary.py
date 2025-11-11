#Example non-uni matrix
non_unitary = np.array([
    [4, 1], 
    [2, 3]
], dtype=complex)



def make_unitary_circuit(matrix):
   
    U, Sig, Vt=np.linalg.svd(matrix)    #Singular value decomposition
    if max(Sig)>1:
        normalize_factor=max(Sig)
        Sig=Sig/normalize_factor
    complex_component=np.sqrt((1-Sig**2)/Sig**2)*Sig  #sigma must be between 0 and 1;  sqrt(eigenvalues) of A*A_conjugate transpose
    sigma_plus=Sig+1j*complex_component
    sigma_minus=Sig-1j*complex_component
    u_sigma=block_diag(np.diag(sigma_plus),np.diag(sigma_minus))    #([I+, 0], [0, I-])
    return u_sigma, Vt, U


u_sig, v_con_trans, unitary=make_unitary_circuit(non_unitary)
u_sig_circuit = UnitaryGate(u_sig, label='U_sigma')
v_trans_circuit=UnitaryGate(v_con_trans, label='Vt')
unitary_circuit= UnitaryGate(unitary, label='U')
unitary_dilation= QuantumCircuit(2)
unitary_dilation.append(v_trans_circuit,[0])
unitary_dilation.h(1)
unitary_dilation.append(u_sig_circuit, [0, 1])
unitary_dilation.h(1)
unitary_dilation.append(unitary_circuit,[0])
unitary_dilation.draw('mpl')
