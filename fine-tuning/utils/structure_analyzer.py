from rdkit import Chem

def find_conjugated_structures(smiles):
    mol = Chem.MolFromSmiles(smiles)
    bond_idx = []
    atom_idx = []
    
    for bond in mol.GetBonds():
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        
        if bond.GetIsConjugated():
            bond_idx.append(bond.GetIdx())
            atom_idx.append(idx1)
            atom_idx.append(idx2)
    
    atom_idx = list(set(atom_idx))
    return bond_idx, atom_idx

def merge_conjugated_substructures(mol, bond_idx, conjugated_atom_idx):
    visited_atoms = set()
    conjugated_substructures = []

    def dfs(atom, current_substructure_atoms, current_substructure_bonds):
        visited_atoms.add(atom.GetIdx())
        current_substructure_atoms.add(atom.GetIdx())
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() in conjugated_atom_idx and neighbor.GetIdx() not in visited_atoms:
                bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                if bond.GetIdx() in bond_idx:
                    current_substructure_bonds.add(bond.GetIdx())
                    dfs(neighbor, current_substructure_atoms, current_substructure_bonds)
    
    for idx in conjugated_atom_idx:
        if idx not in visited_atoms:
            current_substructure_atoms = set()
            current_substructure_bonds = set()
            atom = mol.GetAtomWithIdx(idx)
            dfs(atom, current_substructure_atoms, current_substructure_bonds)
            conjugated_substructures.append({
                "atom_idx": list(current_substructure_atoms),
                "bond_idx": list(current_substructure_bonds)
            })
    
    return conjugated_substructures

def find_skeleton_substructure(conjugated_substructures):
    max_atoms = 0
    skeleton_substructure = None
    
    for substructure in conjugated_substructures:
        num_atoms = len(substructure["atom_idx"])
        if num_atoms > max_atoms:
            max_atoms = num_atoms
            skeleton_substructure = substructure
    
    return skeleton_substructure

def find_side_chains(mol, skeleton):
    skeleton_atoms = set(skeleton['atom_idx'])
    skeleton_bonds = set(skeleton['bond_idx'])
    side_chains = []

    visited_atoms = set()
    
    def dfs(atom, current_side_chain_atoms, current_side_chain_bonds):
        visited_atoms.add(atom.GetIdx())
        current_side_chain_atoms.add(atom.GetIdx())
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() not in skeleton_atoms and neighbor.GetIdx() not in visited_atoms:
                bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                current_side_chain_bonds.add(bond.GetIdx())
                dfs(neighbor, current_side_chain_atoms, current_side_chain_bonds)
    
    for atom_idx in skeleton_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() not in skeleton_atoms and neighbor.GetIdx() not in visited_atoms:
                current_side_chain_atoms = set()
                current_side_chain_bonds = set()
                bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
                current_side_chain_bonds.add(bond.GetIdx())
                dfs(neighbor, current_side_chain_atoms, current_side_chain_bonds)
                side_chains.append({
                    "atom_idx": list(current_side_chain_atoms),
                    "bond_idx": list(current_side_chain_bonds)
                })
    
    return side_chains


def get_structure(smiles):
    mol = Chem.MolFromSmiles(smiles)
    bond_idx, atom_idx = find_conjugated_structures(smiles)
    conjugated_substructures = merge_conjugated_substructures(mol, bond_idx, atom_idx)
    skeleton = find_skeleton_substructure(conjugated_substructures)
    side_chains = find_side_chains(mol, skeleton)
    
    leaves_structure = {
        "skeleton": skeleton,
        "side_chains": side_chains
    }
    
    return leaves_structure


if __name__ == "__main__":
    # 示例使用
    smiles = 'C[C@@H](OC(=O)c1cc(C(N)=O)n(-c2ccccc2)n1)C1CC1'

    mol = Chem.MolFromSmiles(smiles)
    bond_idx, atom_idx = find_conjugated_structures(smiles)
    conjugated_substructures = merge_conjugated_substructures(mol, bond_idx, atom_idx)

    # 找到骨架子结构
    skeleton = find_skeleton_substructure(conjugated_substructures)
    print("Skeleton substructure:")
    print(f"  Atoms (atom_idx): {skeleton['atom_idx']}")
    print(f"  Bonds (bond_idx): {skeleton['bond_idx']}")

    # 找到侧链或取代基
    side_chains = find_side_chains(mol, skeleton)
    print("\nSide chains/substituents:")
    for i, side_chain in enumerate(side_chains):
        print(f"Side Chain {i+1}:")
        print(f"  Atoms (atom_idx): {side_chain['atom_idx']}")
        print(f"  Bonds (bond_idx): {side_chain['bond_idx']}")
