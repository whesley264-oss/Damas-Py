# damas.py
# Regras:
# - 8x8, casas escuras jogáveis.
# - Peão: move 1 para frente (branco sobe, preto desce). Captura em qualquer diagonal.
# - Dama: "voadora" (anda/captura em qualquer distância na diagonal).
# - Captura obrigatória. Múltiplas capturas na mesma peça.
# - Vitória por imobilização ou ausência de peças.

import sys
import math
import time
from typing import List, Tuple, Optional, Dict, Iterable

SIZE = 8
DARK = lambda r, c: (r + c) % 2 == 1

# Peças: 'w','b' peões; 'W','B' damas.
# Turno: 'w' (brancas) primeiro.

Coord = Tuple[int, int]

class Move:
    __slots__ = ("path", "captures", "is_king_move")
    def __init__(self, path: List[Coord], captures: List[Coord]):
        self.path = path              # lista de casas: [(r0,c0)->...->(rn,cn)]
        self.captures = captures      # casas capturadas [(r,c), ...]
        self.is_king_move = False

    def start(self): return self.path[0]
    def end(self):   return self.path[-1]
    def is_capture(self): return len(self.captures) > 0

    def __repr__(self):
        def to_alg(rc):
            r,c = rc
            return chr(ord('a')+c)+str(8-r)
        sep = "x" if self.is_capture() else "-"
        return sep.join(to_alg(p) for p in self.path)

class Board:
    def __init__(self):
        self.sq: List[List[Optional[str]]] = [[None]*SIZE for _ in range(SIZE)]
        for r in range(SIZE):
            for c in range(SIZE):
                if DARK(r,c):
                    if r < 3: self.sq[r][c] = 'b'
                    elif r > 4: self.sq[r][c] = 'w'
        self.turn = 'w'  # branco começa
        self.history: List[str] = []

    def clone(self):
        b = Board.__new__(Board)
        b.sq = [row[:] for row in self.sq]
        b.turn = self.turn
        b.history = self.history[:]
        return b

    def inside(self, r,c): return 0 <= r < SIZE and 0 <= c < SIZE
    def piece(self, rc: Coord): r,c=rc; return self.sq[r][c]
    def setp(self, rc: Coord, v): r,c=rc; self.sq[r][c]=v

    def opposite(self, color): return 'b' if color=='w' else 'w'
    def is_white(self, p): return p in ('w','W')
    def is_black(self, p): return p in ('b','B')
    def color_of(self, p): return 'w' if p in ('w','W') else 'b'
    def is_king(self, p): return p in ('W','B')

    def dirs_simple(self, p:str) -> List[Tuple[int,int]]:
        if self.is_king(p):
            return [(-1,-1),(-1,1),(1,-1),(1,1)]
        return [(-1,-1),(-1,1)] if self.is_white(p) else [(1,-1),(1,1)]

    def dirs_capture(self) -> List[Tuple[int,int]]:
        # Captura pode ser para qualquer diagonal para peões também.
        return [(-1,-1),(-1,1),(1,-1),(1,1)]

    def promote_if_needed(self, rc: Coord):
        r,c = rc
        p = self.piece(rc)
        if not p: return
        if p=='w' and r==0: self.setp(rc,'W')
        if p=='b' and r==7: self.setp(rc,'B')

    def apply(self, mv: Move):
        # Aplica um movimento completo (com múltiplas capturas se houver).
        sr, sc = mv.start()
        er, ec = mv.end()
        p = self.piece((sr,sc))
        self.setp((sr,sc), None)
        self.setp((er,ec), p)
        for cr,cc in mv.captures:
            self.setp((cr,cc), None)
        self.promote_if_needed((er,ec))
        self.turn = self.opposite(self.turn)

    def legal_moves(self, color: str) -> List[Move]:
        # Gera TODOS os movimentos legais do lado "color".
        captures: List[Move] = []
        moves: List[Move] = []
        for r in range(SIZE):
            for c in range(SIZE):
                p = self.sq[r][c]
                if not p: continue
                if (color=='w' and not self.is_white(p)) or (color=='b' and not self.is_black(p)):
                    continue
                cell = (r,c)
                if self.is_king(p):
                    captures.extend(self._king_captures_from(cell))
                    moves.extend(self._king_slides_from(cell))
                else:
                    captures.extend(self._man_captures_from(cell))
                    # movimento simples só 1 passo nas diagonais à frente
                    for dr,dc in self.dirs_simple(p):
                        nr, nc = r+dr, c+dc
                        if self.inside(nr,nc) and DARK(nr,nc) and self.sq[nr][nc] is None:
                            moves.append(Move([cell,(nr,nc)], []))
        # Captura obrigatória
        if captures:
            # expandir múltiplas capturas para caminhos completos
            full_caps = []
            for m in captures:
                full_caps.extend(self._expand_multi_capture(m))
            return full_caps
        return moves

    # --------- geração de movimentos: peão ----------
    def _man_captures_from(self, rc: Coord) -> List[Move]:
        r,c = rc
        p = self.piece(rc)
        res: List[Move] = []
        for dr,dc in self.dirs_capture():
            ar, ac = r+dr, c+dc
            lr, lc = r+2*dr, c+2*dc
            if not (self.inside(lr,lc) and DARK(lr,lc)): continue
            if self.inside(ar,ac) and self.sq[ar][ac] and self.color_of(self.sq[ar][ac]) != self.color_of(p) and self.sq[lr][lc] is None:
                res.append(Move([rc,(lr,lc)], [(ar,ac)]))
        return res

    # expande recursivamente múltiplas capturas para peão
    def _expand_multi_capture(self, m: Move) -> List[Move]:
        # aplica parcialmente e tenta continuar capturando com a MESMA peça
        b = self.clone()
        sr,sc = m.start()
        er,ec = m.end()
        p = b.piece((sr,sc))
        b.setp((sr,sc), None)
        b.setp((er,ec), p)
        for cr,cc in m.captures: b.setp((cr,cc), None)
        # promoção só no final da sequência (padrão comum). Mantemos como peão até terminar.
        cont: List[Move] = []
        if b.is_king(p):
            # Se já era rei, continuar com regras de rei
            next_caps = b._king_captures_from((er,ec))
        else:
            next_caps = b._man_captures_from((er,ec))
        if not next_caps:
            # finaliza e promove se necessário
            if not b.is_king(p):
                if (p=='w' and er==0) or (p=='b' and er==7):
                    m_final = Move(m.path[:], m.captures[:])
                    cont.append(m_final)
                    return cont
            cont.append(m)
            return cont
        for n in next_caps:
            nm = Move(m.path + n.path[1:], m.captures + n.captures)
            cont.extend(self._expand_multi_capture(nm))
        return cont

    # --------- geração de movimentos: dama voadora ----------
    def _king_slides_from(self, rc: Coord) -> List[Move]:
        r,c = rc
        res: List[Move] = []
        for dr,dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = r+dr, c+dc
            while self.inside(nr,nc) and DARK(nr,nc) and self.sq[nr][nc] is None:
                res.append(Move([rc,(nr,nc)], []))
                nr += dr; nc += dc
        for m in res: m.is_king_move = True
        return res

    def _king_captures_from(self, rc: Coord) -> List[Move]:
        r,c = rc
        p = self.piece(rc)
        my = self.color_of(p)
        res: List[Move] = []
        for dr,dc in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = r+dr, c+dc
            # avança por vazios
            while self.inside(nr,nc) and DARK(nr,nc) and self.sq[nr][nc] is None:
                nr += dr; nc += dc
            # encontrou algo?
            if not (self.inside(nr,nc) and DARK(nr,nc)): continue
            if self.sq[nr][nc] and self.color_of(self.sq[nr][nc]) != my:
                # casas de pouso após o inimigo
                lr, lc = nr+dr, nc+dc
                while self.inside(lr,lc) and DARK(lr,lc) and self.sq[lr][lc] is None:
                    res.append(Move([rc,(lr,lc)], [(nr,nc)]))
                    lr += dr; lc += dc
        for m in res: m.is_king_move = True
        # expandir múltiplas capturas de rei
        full: List[Move] = []
        for m in res: full.extend(self._expand_multi_capture(m))
        return full

    # --------- utilidades ----------
    def has_moves(self, color:str) -> bool:
        return len(self.legal_moves(color)) > 0

    def winner(self) -> Optional[str]:
        w_exists = any(p in ('w','W') for row in self.sq for p in row if p)
        b_exists = any(p in ('b','B') for row in self.sq for p in row if p)
        if not w_exists: return 'b'
        if not b_exists: return 'w'
        if not self.has_moves(self.turn): return self.opposite(self.turn)
        return None

    def zobrist(self) -> str:
        # Hash simples stringificado
        rows = []
        for r in range(SIZE):
            rows.append(''.join(self.sq[r][c] if self.sq[r][c] else '.' for c in range(SIZE)))
        return f"{self.turn}|{'/'.join(rows)}"

    def pretty(self):
        # imprime tabuleiro
        s = []
        for r in range(SIZE):
            line = [str(8-r)+" "]
            for c in range(SIZE):
                if not DARK(r,c):
                    line.append(" ")
                    continue
                p = self.sq[r][c]
                line.append(p if p else ".")
            s.append(' '.join(line))
        s.append("  a b c d e f g h")
        return "\n".join(s)

# ---------- Avaliação forte ----------
def evaluate(board: Board) -> float:
    # Positivo favorece 'b' (IA padrão preta). Negativo favorece 'w'.
    score = 0.0
    # Pesos
    W_P, W_K = 3.0, 6.0           # material
    W_ADV, W_CTR = 0.2, 0.15      # avanço e centro
    W_MOB = 0.05                  # mobilidade
    W_PROMO = 0.6                 # quase promoção
    W_HANG = 0.7                  # peça pendurada
    center = {(3,3),(3,5),(4,2),(4,4)}
    # material + posição
    for r in range(SIZE):
        for c in range(SIZE):
            p = board.sq[r][c]
            if not p: continue
            val = W_K if p in ('W','B') else W_P
            adv = 0.0
            if p in ('b','B'): adv += r*W_ADV
            if p in ('w','W'): adv += (7-r)*W_ADV
            ctr = W_CTR if (r,c) in center else 0.0
            v = val + adv + ctr
            score += v if p in ('b','B') else -v
            # bônus por quase-promoção
            if p=='b' and r==6: score += W_PROMO
            if p=='w' and r==1: score -= W_PROMO
    # mobilidade
    bm = len(board.legal_moves('b'))
    wm = len(board.legal_moves('w'))
    score += W_MOB * (bm - wm)

    # penaliza peças penduradas: se oponente tem captura imediata de mim
    score -= W_HANG * count_hanging(board, 'b')
    score += W_HANG * count_hanging(board, 'w')  # se brancas penduradas, favorece preto

    return score

def count_hanging(b: Board, color: str) -> int:
    # conta quantas peças de 'color' podem ser capturadas imediatamente
    opp = b.opposite(color)
    caps = b.legal_moves(opp)
    return sum(1 for m in caps for _ in m.captures)

# ---------- Minimax + alfa-beta + TT ----------
TT: Dict[str, Tuple[int,float]] = {}  # hash -> (depth, eval)

def negamax(b: Board, depth: int, alpha: float, beta: float, color_sign: int) -> float:
    key = b.zobrist()
    if key in TT and TT[key][0] >= depth:
        return TT[key][1]

    win = b.winner()
    if win is not None:
        val = 1e6 if win=='b' else -1e6
        TT[key] = (depth, val)
        return val

    if depth == 0:
        val = color_sign * evaluate(b)
        TT[key] = (depth, val)
        return val

    moves = b.legal_moves(b.turn)
    if not moves:
        # sem movimentos: perdeu
        val = -1e6
        TT[key] = (depth, val)
        return val

    # ordenação: capturas e promoções primeiro
    def move_order(m: Move):
        sr,sc = m.start(); er,ec = m.end()
        p = b.piece((sr,sc))
        promo = 1 if (p=='b' and er==7) or (p=='w' and er==0) else 0
        return (m.is_capture(), promo, -len(m.captures))
    moves.sort(key=move_order, reverse=True)

    best = -math.inf
    for m in moves:
        nb = b.clone()
        nb.apply(m)
        val = -negamax(nb, depth-1, -beta, -alpha, -color_sign)
        if val > best: best = val
        if best > alpha: alpha = best
        if alpha >= beta: break
    TT[key] = (depth, best)
    return best

def best_move(b: Board, depth: int) -> Move:
    moves = b.legal_moves(b.turn)
    if not moves: raise RuntimeError("Sem movimentos")
    alpha, beta = -math.inf, math.inf
    best = moves[0]; best_val = -math.inf
    # ordenação inicial
    moves.sort(key=lambda m: (m.is_capture(), -len(m.captures)), reverse=True)
    for m in moves:
        nb = b.clone()
        nb.apply(m)
        val = -negamax(nb, depth-1, -beta, -alpha, -1 if b.turn=='w' else 1)
        if val > best_val:
            best_val = val; best = m
        if best_val > alpha: alpha = best_val
    return best

# ---------- I/O ----------
def parse_move(txt: str, board: Board) -> Optional[Move]:
    # formatos: a3-b4  ou  a3-c5-e7  (capturas múltiplas)
    txt = txt.strip().lower().replace('x','-')
    parts = [p for p in txt.split('-') if p]
    if len(parts) < 2: return None
    def to_rc(s):
        if len(s)!=2: return None
        c = ord(s[0]) - ord('a')
        r = 8 - int(s[1])
        if not (0<=r<8 and 0<=c<8 and DARK(r,c)): return None
        return (r,c)
    path = []
    for p in parts:
        rc = to_rc(p)
        if rc is None: return None
        path.append(rc)
    # Encontrar move legal com esse path
    legal = board.legal_moves(board.turn)
    for m in legal:
        if m.path == path:
            return m
    # permitir usuário informar só início e fim para movimento simples
    for m in legal:
        if not m.is_capture() and len(m.path)==2 and m.path[0]==path[0] and m.path[-1]==path[-1]:
            return m
    return None

def list_legal(board: Board):
    moves = board.legal_moves(board.turn)
    if not moves:
        print("Sem movimentos.")
        return
    print(f"Jogadas legais ({'pretas' if board.turn=='b' else 'brancas'}):")
    for i,m in enumerate(moves,1):
        print(f"{i:2d}. {m}")

def game_loop():
    print("WS⚙️😈 Damas | Modos: 1) Humano vs IA  2) Humano vs Humano  3) IA vs IA")
    mode = input("Escolha modo [1/2/3]: ").strip() or "1"
    depth = int(input("Profundidade IA (sugestão 6): ").strip() or "6")
    b = Board()
    ai_side = 'b' if mode=="1" else None
    print("\nDigite 'help' para ajuda, 'moves' para listar lances, 'quit' para sair.\n")
    while True:
        print(b.pretty())
        win = b.winner()
        if win:
            print(f"Vencedor: {'pretas' if win=='b' else 'brancas'}")
            return
        turn_name = "pretas" if b.turn=='b' else "brancas"
        if (mode=="3") or (ai_side==b.turn):
            # IA joga
            t0=time.time()
            mv = best_move(b, depth)
            t1=time.time()
            print(f"IA ({turn_name}) joga: {mv}  [{t1-t0:.2f}s]")
            b.apply(mv)
            continue
        # Humano
        cmd = input(f"Sua vez ({turn_name}) > ").strip().lower()
        if cmd in ("quit","exit"): return
        if cmd=="help":
            print("Formato: a3-b4 para movimento simples; a3-c5-e7 para múltiplas capturas.")
            print("Use 'moves' para listar jogadas legais.")
            continue
        if cmd=="moves":
            list_legal(b); continue
        mv = parse_move(cmd, b)
        if not mv:
            print("Lance inválido. Use 'moves' para ver opções.")
            continue
        b.apply(mv)

if __name__ == "__main__":
    try:
        game_loop()
    except KeyboardInterrupt:
        print("\nEncerrado.")
