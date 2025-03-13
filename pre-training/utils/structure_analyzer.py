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
    """
    Find all side chains connected to the skeleton
    Args:
        mol: RDKit molecule object
        skeleton: Dictionary containing atom_idx and bond_idx of the skeleton
    Returns:
        list: List of dictionaries containing atom_idx and bond_idx for each side chain
    """
    skeleton_atoms = set(skeleton['atom_idx'])
    side_chains = []
    
    def dfs(start_atom, visited, current_bonds):
        """
        Depth-first search to find connected atoms in the side chain
        Args:
            start_atom: Current atom being visited
            visited: Set of visited atom indices
            current_bonds: Set of bonds in the current side chain
        """
        visited.add(start_atom.GetIdx())
        
        for neighbor in start_atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in skeleton_atoms and neighbor_idx not in visited:
                # Add the bond to the side chain
                bond = mol.GetBondBetweenAtoms(start_atom.GetIdx(), neighbor_idx)
                current_bonds.add(bond.GetIdx())
                # Continue DFS
                dfs(neighbor, visited, current_bonds)
    
    # Find all attachment points from skeleton to side chains
    for atom_idx in skeleton_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        
        for neighbor in atom.GetNeighbors():
            if neighbor.GetIdx() not in skeleton_atoms:
                # Start a new side chain
                current_atoms = set()
                current_bonds = set()
                
                # Add the connecting bond
                bond = mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx())
                current_bonds.add(bond.GetIdx())
                
                # Add the first non-skeleton atom
                current_atoms.add(neighbor.GetIdx())
                
                # Find the rest of the side chain
                dfs(neighbor, current_atoms, current_bonds)
                
                # Add the side chain to the list
                if current_atoms:  # Only add if we found atoms
                    side_chains.append({
                        "atom_idx": list(current_atoms),
                        "bond_idx": list(current_bonds)
                    })
    # 合并具有相同atom_idx的side chains
    merged_side_chains = []
    atom_idx_dict = {}
    
    # 遍历所有side chains,按atom_idx分组
    for chain in side_chains:
        atom_key = tuple(sorted(chain['atom_idx']))  # 转换成tuple作为字典key
        if atom_key in atom_idx_dict:
            # 如果已存在相同的atom_idx,合并bond_idx
            existing_chain = atom_idx_dict[atom_key]
            existing_chain['bond_idx'] = list(set(existing_chain['bond_idx'] + chain['bond_idx']))
        else:
            # 如果是新的atom_idx组合,添加到字典
            atom_idx_dict[atom_key] = {
                'atom_idx': list(atom_key),
                'bond_idx': chain['bond_idx']
            }
    
    # 将合并后的结果转换为列表
    side_chains = list(atom_idx_dict.values())
    
    return side_chains




def find_largest_fused_ring(mol):
    """
    Find the largest fused ring system in a molecule
    Args:
        mol: RDKit molecule object
    Returns:
        dict: Dictionary containing atom and bond indices of the largest fused ring system
    """
    ri = mol.GetRingInfo()
    rings = ri.AtomRings()
    if not rings:
        return None
    
    visited_atoms = set()
    max_fused_system = []
    
    def get_fused_rings(ring_idx):
        ring_system = list(rings[ring_idx])
        visited_rings = {ring_idx}
        
        for i in range(len(rings)):
            if i not in visited_rings:
                ring = set(rings[i])
                if ring.intersection(set(ring_system)):
                    ring_system.extend(list(ring - set(ring_system)))
                    visited_rings.add(i)
        
        return ring_system
    
    for i in range(len(rings)):
        if i not in visited_atoms:
            current_system = get_fused_rings(i)
            if len(current_system) > len(max_fused_system):
                max_fused_system = current_system
    
    skeleton_bonds = set()
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx in max_fused_system and end_idx in max_fused_system:
            skeleton_bonds.add(bond.GetIdx())
    
    return {
        "atom_idx": list(max_fused_system),
        "bond_idx": list(skeleton_bonds)
    }

def find_largest_conjugated_ring(mol):
    """
    Find the largest conjugated ring in a molecule
    Args:
        mol: RDKit molecule object
    Returns:
        dict: Dictionary containing atom and bond indices of the largest conjugated ring
    """
    ri = mol.GetRingInfo()
    rings = ri.AtomRings()
    if not rings:
        return None
    
    max_conjugated_ring = []
    max_conjugated_bonds = set()
    
    for ring in rings:
        is_conjugated = True
        ring_bonds = set()
        
        for i in range(len(ring)):
            atom1 = ring[i]
            atom2 = ring[(i + 1) % len(ring)]
            bond = mol.GetBondBetweenAtoms(atom1, atom2)
            if not bond.GetIsConjugated():
                is_conjugated = False
                break
            ring_bonds.add(bond.GetIdx())
        
        if is_conjugated and len(ring) > len(max_conjugated_ring):
            max_conjugated_ring = list(ring)
            max_conjugated_bonds = ring_bonds
    
    return {
        "atom_idx": max_conjugated_ring,
        "bond_idx": list(max_conjugated_bonds)
    }
    
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

def PartitionMol(smiles, bone_selection="lgcs"):
    """
    Partition a molecule into skeleton and side chains
    Args:
        smiles: SMILES string of the molecule
        bone_selection: Method to select the skeleton
            - lgfr: Largest Fused Ring
            - lgcr: Largest Conjugated Ring
            - lgcs: Largest Conjugated System
    Returns:
        dict: Dictionary containing skeleton and side chains information
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if bone_selection.lower() == "lgcs":
        return get_structure(smiles)
    
    elif bone_selection.lower() == "lgfr":
        skeleton = find_largest_fused_ring(mol)
        
    elif bone_selection.lower() == "lgcr":
        skeleton = find_largest_conjugated_ring(mol)
    
    else:
        raise ValueError(f"Unsupported bone selection method: {bone_selection}")
    
    if skeleton is None:
        return None
        
    side_chains = find_side_chains(mol, skeleton)
    
    return {
        "skeleton": skeleton,
        "side_chains": side_chains
    }

if __name__ == "__main__":
    # 示例使用
    smiles = 'O=C(Nc1cccc(C(=O)OCc2ccccc2)c1)c1ccc(CO)cc1'

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
