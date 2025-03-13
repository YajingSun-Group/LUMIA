from rdkit import Chem

def find_conjugated_bonds_and_atoms(mol):
    """
    识别分子中的所有共轭和芳香键。

    返回值：
    - conjugated_bonds: 共轭或芳香键的索引列表。
    - conjugated_atoms: 参与共轭或芳香键的原子索引集合。
    """
    conjugated_bonds = []
    conjugated_atoms = set()

    for bond in mol.GetBonds():
        if bond.GetIsConjugated() or bond.GetIsAromatic():
            conjugated_bonds.append(bond.GetIdx())
            conjugated_atoms.add(bond.GetBeginAtomIdx())
            conjugated_atoms.add(bond.GetEndAtomIdx())

    return conjugated_bonds, conjugated_atoms

def build_conjugated_substructures(mol, conjugated_bonds, conjugated_atoms):
    """
    构建连通的共轭子结构。

    返回值：
    - conjugated_substructures: 每个子结构的字典列表，包含 'atom_idx' 和 'bond_idx'。
    """
    # 为共轭原子构建邻接列表
    adj_list = {}
    for bond_idx in conjugated_bonds:
        bond = mol.GetBondWithIdx(bond_idx)
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        adj_list.setdefault(idx1, set()).add(idx2)
        adj_list.setdefault(idx2, set()).add(idx1)

    visited = set()
    conjugated_substructures = []

    for atom_idx in conjugated_atoms:
        if atom_idx not in visited:
            current_atoms = set()
            current_bonds = set()
            stack = [atom_idx]
            visited.add(atom_idx)
            current_atoms.add(atom_idx)

            while stack:
                current_atom_idx = stack.pop()
                neighbors = adj_list.get(current_atom_idx, set())
                for neighbor_idx in neighbors:
                    bond = mol.GetBondBetweenAtoms(current_atom_idx, neighbor_idx)
                    bond_idx = bond.GetIdx()
                    if neighbor_idx not in visited:
                        visited.add(neighbor_idx)
                        stack.append(neighbor_idx)
                        current_atoms.add(neighbor_idx)
                        current_bonds.add(bond_idx)
                    else:
                        current_bonds.add(bond_idx)

            conjugated_substructures.append({
                'atom_idx': list(current_atoms),
                'bond_idx': list(current_bonds)
            })

    return conjugated_substructures

def select_largest_conjugated_substructure(conjugated_substructures):
    """
    选择最大的共轭子结构作为骨架。

    返回值：
    - skeleton: 包含骨架的 'atom_idx' 和 'bond_idx' 的字典。
    """
    if not conjugated_substructures:
        return {'atom_idx': [], 'bond_idx': []}

    # 选择包含最多原子的子结构
    skeleton = max(conjugated_substructures, key=lambda x: len(x['atom_idx']))
    return skeleton

def find_side_chains(mol, skeleton_atoms):
    """
    识别附着在骨架上的侧链。

    返回值：
    - side_chains: 每个侧链的字典列表，包含 'atom_idx' 和 'bond_idx'。
    """
    skeleton_atoms = set(skeleton_atoms)
    side_chains = []
    visited_atoms = set(skeleton_atoms)

    for atom_idx in skeleton_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx not in visited_atoms:
                current_side_chain_atoms = set()
                current_side_chain_bonds = set()
                stack = [neighbor]
                visited_atoms.add(neighbor_idx)
                current_side_chain_atoms.add(neighbor_idx)

                while stack:
                    current_atom = stack.pop()
                    for nbr in current_atom.GetNeighbors():
                        nbr_idx = nbr.GetIdx()
                        bond = mol.GetBondBetweenAtoms(current_atom.GetIdx(), nbr_idx)
                        bond_idx = bond.GetIdx()
                        if nbr_idx not in skeleton_atoms and nbr_idx not in visited_atoms:
                            visited_atoms.add(nbr_idx)
                            stack.append(nbr)
                            current_side_chain_atoms.add(nbr_idx)
                            current_side_chain_bonds.add(bond_idx)
                        elif nbr_idx in skeleton_atoms:
                            current_side_chain_bonds.add(bond_idx)

                side_chains.append({
                    'atom_idx': list(current_side_chain_atoms),
                    'bond_idx': list(current_side_chain_bonds)
                })

    return side_chains

def get_structure(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # 无效的 SMILES

    # 识别共轭键和原子
    conjugated_bonds, conjugated_atoms = find_conjugated_bonds_and_atoms(mol)

    # 构建连通的共轭子结构
    conjugated_substructures = build_conjugated_substructures(mol, conjugated_bonds, conjugated_atoms)

    # 选择最大的连续π共轭骨架
    skeleton = select_largest_conjugated_substructure(conjugated_substructures)

    # 如果未找到共轭骨架，使用最长链作为骨架
    if not skeleton['atom_idx']:
        skeleton = find_longest_chain_skeleton(mol)

    # 识别侧链
    side_chains = find_side_chains(mol, skeleton['atom_idx'])

    leaves_structure = {
        "skeleton": skeleton,
        "side_chains": side_chains
    }

    return leaves_structure

def find_longest_chain_skeleton(mol):
    """
    如果不存在共轭系统，找到最长的链作为骨架。

    返回值：
    - skeleton: 包含骨架的 'atom_idx' 和 'bond_idx' 的字典。
    """
    from rdkit.Chem.rdmolops import GetShortestPath

    max_length = 0
    skeleton_atoms = []
    skeleton_bonds = []

    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        return {'atom_idx': [], 'bond_idx': []}

    for atom1 in mol.GetAtoms():
        for atom2 in mol.GetAtoms():
            if atom1.GetIdx() >= atom2.GetIdx():
                continue
            path = GetShortestPath(mol, atom1.GetIdx(), atom2.GetIdx())
            if len(path) > max_length:
                max_length = len(path)
                skeleton_atoms = path

    for i in range(len(skeleton_atoms) - 1):
        bond = mol.GetBondBetweenAtoms(skeleton_atoms[i], skeleton_atoms[i + 1])
        skeleton_bonds.append(bond.GetIdx())

    skeleton = {
        'atom_idx': list(skeleton_atoms),
        'bond_idx': skeleton_bonds
    }
    return skeleton

if __name__ == "__main__":
    # 示例使用
    smiles_list = [
        'O=C(Nc1ccccc1)C(=O)Nc1ccc2oc(C(=O)O)cc2c1',  # 原始示例
        'Cc1ccccc1',  # 甲苯
        'c1ccncc1',   # 吡啶
        'c1ccc2ccccc2c1',  # 萘
        'CC(C)=CC=C',  # 异戊二烯
        'C1=CC=CC=C1',  # 苯
        'CCOC(=O)c1ccccc1',  # 苯甲酸乙酯
        'CC(=O)NC1=CC=CC=C1',  # 乙酰苯胺
        'c1ccco1',  # 呋喃
        'c1ccncn1',  # 嘧啶
        'CCOC(=O)C1=CC=CC=C1',  # 肉桂酸乙酯
        'CC(=O)C=CC=O',  # 戊-2-烯-4-酮
    ]

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        structure = get_structure(smiles)

        if structure is None:
            print(f"Invalid SMILES: {smiles}\n")
            continue

        skeleton = structure['skeleton']
        side_chains = structure['side_chains']

        print(f"SMILES: {smiles}")
        print("Skeleton substructure:")
        print(f"  Atoms (atom_idx): {sorted(skeleton['atom_idx'])}")
        print(f"  Bonds (bond_idx): {sorted(skeleton['bond_idx'])}")

        print("\nSide chains/substituents:")
        for i, side_chain in enumerate(side_chains):
            print(f"  Side Chain {i+1}:")
            print(f"    Atoms (atom_idx): {sorted(side_chain['atom_idx'])}")
            print(f"    Bonds (bond_idx): {sorted(side_chain['bond_idx'])}")
        print("\n" + "-"*50 + "\n")
