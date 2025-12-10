# Compilador BNVCode
# Integrantes:
# - Bryan Misael Morales Martin
# - Naomi Hernandez Romo
# - Ricardo Andres Veloz Hernandez
# 9A TM 

# Importar librerías necesarias, incluyendo:
import sys 
import os # Para manejar rutas de archivos
import re # Para las expresiones regulares del analizador léxico

# Y todos los modulos de PyQt5 necesarios para la interfaz gráfica
from PyQt5.QtCore import QRegularExpression, Qt, QProcess
from PyQt5.QtGui import (
    QSyntaxHighlighter,  
    QTextCharFormat, 
    QColor, 
    QFont,
    QPainter,
    QTextCursor
)
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QThread, pyqtSignal

class PInterpreterThread(QThread):
    """Hilo para ejecutar el intérprete de código sin bloquear la interfaz"""
    output_signal = pyqtSignal(str)
    input_request_signal = pyqtSignal(str)
    execution_finished = pyqtSignal()
    error_signal = pyqtSignal(str)
    
    def __init__(self, code_text):
        super().__init__()
        self.code_text = code_text
        self.input_value = None
        self.waiting_for_input = False
        self.should_stop = False
        self.interpreter = None
        
    def run(self):
        """Ejecuta el intérprete en un hilo separado"""
        try:
            self.interpreter = PInterpreter(self.code_text, self)
            self.interpreter.run()
            self.execution_finished.emit()
        except Exception as e:
            self.error_signal.emit(f"Error en ejecución: {str(e)}")
    
    def provide_input(self, value):
        """Provee entrada al intérprete cuando lo solicita"""
        self.input_value = value
        self.waiting_for_input = False

class PInterpreter:
    """Intérprete de código con soporte completo y parsing corregido"""
    
    def __init__(self, code_text, thread):
        self.code_text = code_text
        self.thread = thread
        self.instructions = []
        self.parse_code()
        
        # Estado del intérprete
        self.registers = [0] * 8  # Registros 0-7
        self.memory = [0] * 10000  # Memoria principal
        self.pc = 0  # Contador de programa
        self.running = False
        self.output_buffer = []
        self.input_buffer = []
        
        # Inicializar registros importantes
        self.registers[5] = 0  # Base para variables globales
        self.registers[6] = 0  # Puntero de pila
        self.registers[7] = 0  # Registro de retorno
        
        self.next_output_type = None  # 'string', 'number', o None
        
    def parse_code(self):
        """Parsea el código en instrucciones"""
        lines = self.code_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Extraer instrucción (omitir número de línea si existe)
            if ':' in line:
                # Formato: "  0:    LD  6,0(0)"
                instr_part = line.split(':', 1)[1].strip()
            else:
                instr_part = line
                
            if instr_part:
                self.instructions.append(instr_part)
    
    def run(self):
        """Ejecuta el programa P"""
        self.running = True
        self.pc = 0
        
        while self.running and self.pc < len(self.instructions) and not self.thread.should_stop:
            self.execute_instruction(self.instructions[self.pc])
        
        self.running = False
    
    def parse_instruction_parts(self, instr):
        """Parsea una instrucción en sus componentes"""
        parts = instr.strip().split()
        if not parts:
            return None, []
        
        opcode = parts[0]
        
        # Manejar instrucciones sin operandos
        if len(parts) == 1:
            return opcode, []
        
        # Para instrucciones con operandos, manejar diferentes formatos
        operands = []
        
        if opcode in ['LDC', 'LD', 'ST', 'LDA']:
            # Formato: OP reg,offset(base)
            if ',' in parts[1]:
                reg_part, addr_part = parts[1].split(',', 1)
                reg = int(reg_part.strip())
                
                # Parsear offset(base)
                if '(' in addr_part and ')' in addr_part:
                    offset_str, base_str = addr_part.split('(', 1)
                    offset = int(offset_str.strip())
                    base_reg = int(base_str.strip(')'))
                    operands = [reg, offset, base_reg]
                else:
                    # Solo offset, sin base
                    offset = int(addr_part.strip())
                    operands = [reg, offset, 0]
            else:
                # Formato alternativo: OP reg offset base
                if len(parts) >= 4:
                    reg = int(parts[1].strip())
                    offset = int(parts[2].strip())
                    base = int(parts[3].strip('()'))
                    operands = [reg, offset, base]
        
        elif opcode in ['ADD', 'SUB', 'MUL', 'DIV', 'MOD']:
            # Formato: OP dest,src1,src2
            if ',' in parts[1]:
                ops = parts[1].split(',')
                if len(ops) == 3:
                    operands = [int(op.strip()) for op in ops]
            elif len(parts) >= 4:
                operands = [int(parts[1].strip()), int(parts[2].strip()), int(parts[3].strip())]
        
        elif opcode in ['JLT', 'JLE', 'JGT', 'JGE', 'JEQ', 'JNE']:
            # Formato: OP reg,offset(base) o a veces OP reg,basura,offset(base)
            if ',' in parts[1]:
                all_ops = parts[1].split(',')
                reg_part = all_ops[0]
                offset_part = all_ops[-1]  # Tomar la última parte para el offset
                
                reg = int(reg_part.strip())
                offset = int(offset_part.split('(')[0].strip())
                operands = [reg, offset]
            elif len(parts) >= 3:
                reg = int(parts[1].strip())
                offset = int(parts[2].split('(')[0].strip())
                operands = [reg, offset]
        
        elif opcode in ['IN', 'OUT']:
            # Formato: OP reg,0,0
            if ',' in parts[1]:
                reg_part = parts[1].split(',')[0]
                reg = int(reg_part.strip())
                operands = [reg]
            elif len(parts) >= 2:
                reg = int(parts[1].strip())
                operands = [reg]
        
        elif opcode == 'HALT':
            operands = []
        
        return opcode, operands
    
    def execute_instruction(self, instr):
        """Ejecuta una instrucción individual"""
        opcode, operands = self.parse_instruction_parts(instr)
        
        if opcode is None:
            self.pc += 1
            return
        
        try:
            # LDC: Cargar constante
            if opcode == 'LDC':
                if len(operands) >= 2:
                    reg = operands[0]
                    value = operands[1]
                    self.registers[reg] = value
                self.pc += 1
            
            # LD: Cargar desde memoria
            elif opcode == 'LD':
                if len(operands) >= 3:
                    reg = operands[0]
                    offset = operands[1]
                    base_reg = operands[2]
                    effective_addr = offset + self.registers[base_reg]
                    
                    if 0 <= effective_addr < len(self.memory):
                        self.registers[reg] = self.memory[effective_addr]
                    else:
                        self.thread.error_signal.emit(f"Dirección de memoria inválida: {effective_addr}")
                        self.running = False
                self.pc += 1
            
            # ST: Almacenar en memoria
            elif opcode == 'ST':
                if len(operands) >= 3:
                    reg = operands[0]
                    offset = operands[1]
                    base_reg = operands[2]
                    effective_addr = offset + self.registers[base_reg]
                    
                    if 0 <= effective_addr < len(self.memory):
                        self.memory[effective_addr] = self.registers[reg]
                    else:
                        self.thread.error_signal.emit(f"Dirección de memoria inválida: {effective_addr}")
                        self.running = False
                self.pc += 1
            
            # ADD: Sumar
            elif opcode == 'ADD':
                if len(operands) >= 3:
                    dest, src1, src2 = operands
                    self.registers[dest] = self.registers[src1] + self.registers[src2]
                self.pc += 1
            
            # SUB: Restar
            elif opcode == 'SUB':
                if len(operands) >= 3:
                    dest, src1, src2 = operands
                    self.registers[dest] = self.registers[src1] - self.registers[src2]
                self.pc += 1
            
            # MUL: Multiplicar
            elif opcode == 'MUL':
                if len(operands) >= 3:
                    dest, src1, src2 = operands
                    self.registers[dest] = self.registers[src1] * self.registers[src2]
                self.pc += 1
            
            # DIV: Dividir
            elif opcode == 'DIV':
                if len(operands) >= 3:
                    dest, src1, src2 = operands
                    if self.registers[src2] != 0:
                        # Usar división flotante en lugar de entera
                        self.registers[dest] = self.registers[src1] / self.registers[src2]
                    else:
                        self.thread.error_signal.emit("División por cero")
                        self.running = False
                self.pc += 1
            
            # MOD: Módulo
            elif opcode == 'MOD':
                if len(operands) >= 3:
                    dest, src1, src2 = operands
                    if self.registers[src2] != 0:
                        self.registers[dest] = self.registers[src1] % self.registers[src2]
                    else:
                        self.thread.error_signal.emit("Módulo por cero")
                        self.running = False
                self.pc += 1
            
            # Instrucciones de salto
            elif opcode in ['JLT', 'JLE', 'JGT', 'JGE', 'JEQ', 'JNE']:
                if len(operands) >= 2:
                    reg, offset = operands
                    condition = False
                    
                    if opcode == 'JLT':
                        condition = self.registers[reg] < 0
                    elif opcode == 'JLE':
                        condition = self.registers[reg] <= 0
                    elif opcode == 'JGT':
                        condition = self.registers[reg] > 0
                    elif opcode == 'JGE':
                        condition = self.registers[reg] >= 0
                    elif opcode == 'JEQ':
                        condition = self.registers[reg] == 0
                    elif opcode == 'JNE':
                        condition = self.registers[reg] != 0
                    
                    if condition:
                        self.pc += offset
                    else:
                        self.pc += 1
                else:
                    self.pc += 1
            
            # LDA: Cargar dirección
            elif opcode == 'LDA':
                if len(operands) >= 3:
                    reg, offset, base_reg = operands
                    self.registers[reg] = offset + self.registers[base_reg]
                elif len(operands) == 2:
                    reg, offset = operands
                    self.registers[reg] = offset
                self.pc += 1
            
            # IN: Leer entrada
            elif opcode == 'IN':
                reg = operands[0] if operands else 0
                
                if not self.input_buffer:
                    # Si el buffer está vacío, solicitar nueva entrada
                    self.thread.input_request_signal.emit("Ingrese un valor:")
                    self.thread.waiting_for_input = True
                    
                    # Esperar por entrada
                    while self.thread.waiting_for_input and not self.thread.should_stop:
                        self.thread.msleep(100)
                    
                    if self.thread.input_value is not None:
                        # Dividir la entrada y guardarla en el buffer
                        self.input_buffer.extend(self.thread.input_value.strip().split())
                
                # Si todavía no hay nada en el buffer (ej. entrada vacía), usar 0
                if not self.input_buffer:
                    self.registers[reg] = 0
                else:
                    # Tomar el siguiente valor del buffer
                    input_val = self.input_buffer.pop(0)
                    try:
                        # Intentar convertir a float primero
                        try:
                            self.registers[reg] = float(input_val)
                        except ValueError:
                            # Si falla, intentar como int
                            self.registers[reg] = int(input_val)
                    except ValueError:
                        self.thread.error_signal.emit(f"Valor inválido: {input_val}")
                        self.registers[reg] = 0
                
                self.pc += 1
            
            # OUT: Escribir salida
            elif opcode == 'OUT':
                reg = operands[0] if operands else 0
                value = self.registers[reg]
                
                # Manejar códigos especiales
                if self.next_output_type is not None:
                    if self.next_output_type == 'string':
                        # Mostrar como carácter
                        output = self.format_as_char(value)
                    else:  # 'number'
                        # Mostrar como número
                        output = str(value) + " "
                    
                    self.output_buffer.append(output)
                    self.thread.output_signal.emit(output)
                    self.next_output_type = None
                elif value in [0, 1]:  # Códigos especiales
                    if value == 0:
                        self.next_output_type = 'string'
                    else:  # value == 1
                        self.next_output_type = 'number'
                else:
                    # Para compatibilidad con código viejo
                    output = self.format_output(value)
                    self.output_buffer.append(output)
                    self.thread.output_signal.emit(output)
                
                self.pc += 1
            
            # HALT: Terminar ejecución
            elif opcode == 'HALT':
                self.running = False
                self.thread.output_signal.emit("\n")
            
            else:
                self.thread.error_signal.emit(f"Opcode desconocido: {opcode}")
                self.running = False
        
        except Exception as e:
            self.thread.error_signal.emit(f"Error ejecutando instrucción '{instr}': {str(e)}")
            self.running = False
    
    def format_as_char(self, value):
        """Formatea un valor como carácter ASCII"""
        if isinstance(value, (int, float)):
            if value == 10:  # Salto de línea
                return "\n"
            elif value == 13:  # Retorno de carro
                return ""
            elif value == 9:   # Tabulación
                return "\t"
            elif value == 32:  # Espacio 
                return " "
            elif 32 <= value <= 126:  # Caracteres imprimibles
                return chr(int(value))
            else:
                # Si no es imprimible, mostrar como número
                if isinstance(value, float):
                    return f"{value:.2f} "
                else:
                    return str(value) + " "  
        else:
            return str(value)
    
    def format_output(self, value):
        """Formatea un valor para salida, manejando números negativos, ASCII, etc."""
        # Si es un valor negativo codificado (bit más alto activado)
        if isinstance(value, (int, float)):
            if value >= 32768:
                # Convertir de pseudo-complemento a 2 (valor codificado)
                actual_value = value - 65536
                return str(actual_value) + " "
            elif value < 0:
                return str(value) + " "
            elif 32 <= value <= 126:
                return chr(int(value))
            elif value == 10:  # Salto de línea
                return "\n"
            elif value == 13:  # Retorno de carro
                return ""
            elif value == 9:   # Tabulación
                return "\t"
            elif value == 32:  # Espacio
                return " "
            else:
                # Para números enteros o flotantes
                if isinstance(value, float):
                    # Mostrar con precisión decimal
                    return f"{value:.2f} "
                else:
                    return str(value) + " "
        else:
            return str(value) + " "

# Definición de la clase LexicalAnalyzer

# Esta clase se encarga de analizar el código fuente y generar tokens
# y errores léxicos. Utiliza expresiones regulares para identificar
# diferentes tipos de tokens, como comentarios, palabras reservadas,
# números, identificadores, operadores y símbolos. También maneja
# errores léxicos y genera una lista de errores encontrados durante
# el análisis.

class LexicalAnalyzer:
    # Definición de los patrones de tokens
    # Cada patrón tiene un nombre, una expresión regular y un color
    # para la representación visual en el editor de texto.
    TOKEN_SPEC = [
        # Comentarios unilínea y multi-línea
        ('COMENTARIO_MULTI', r'/\*[\s\S]*?\*/', QColor('#6A9955')),  # Usando [\s\S] para incluir saltos
        ('COMENTARIO_LINEA', r'//.*', QColor('#6A9955')),
        # Palabras reservadas
        ('PALABRA_RESERVADA', r'\b(if|then|else|end|do|until|while|switch|case|int|float|main|cin|cout|bool)\b', QColor('#569CD6')),
        # Booleanos
        ('BOOLEANO', r'\b(true|false)\b', QColor('#569CD6')),
        # Números
        ('NUM_REAL_INCOMPLETO', r'\d+\.(?!\d)', QColor('#FF0000')), # Errores en números reales
        ('NUM_REAL', r'\d+\.\d+\b', QColor('#1788ff')),
        ('NUM_ENTERO', r'\d+\b', QColor('#1788ff')),
        # Identificadores
        ('IDENTIFICADOR', r'[a-zA-Z_][a-zA-Z0-9_]*', QColor('#ff56e3')),
        # Operadores y símbolos
        ('OPERADOR_ARIT', r'(\+\+|--|\+|-|\*|/|%|\^)', QColor('#DCDCAA')),
        ('OPERADOR_ES', r'(>>|<<)', QColor('#DCDCAA')),
        ('OPERADOR_REL', r'(<=|>=|==|!=|<|>)', QColor('#DCDCAA')),
        ('OPERADOR_LOG', r'(\&\&|\|\||!)', QColor('#DCDCAA')),
        ('ASIGNACION', r'=', QColor('#DCDCAA')),
        # Cadenas de texto
        ('CADENA_DOBLE', r'"(?:\\.|[^"\\])*"', QColor('#CE9178')),
        ('CADENA_SIMPLE', r"'(?:\\.|[^'\\])*'", QColor('#CE9178')),
        # Simbolos
        ('COMA', r',', QColor('#FFD700')),
        ('SIMBOLOS', r'[\(\)\[\]\{\};]', QColor('#FFD700')),
        # Errores
        ('ERROR', r'.', QColor('#FF0000')) # Captura errores
    ]
    
    # Inicializando el analizador léxico
    def __init__(self):
        self.tokens = [] # Lista para almacenar los tokens generados
        self.errors = [] # Lista para almacenar los errores léxicos
        
        # Compilando la expresión regular para el análisis léxico
        # Uniendo los patrones de tokens en una sola expresión regular
        self.regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern, _ in self.TOKEN_SPEC)
    
    # Método para analizar el código fuente
    def analyze(self, code):
        # Se limpian ambas listas (tokens y errores)
        self.tokens.clear()
        self.errors.clear()
        
        # Se inicializan punteros para el analisis
        last_pos = 0
        line_num = 1
        line_start = 0
        
        # Se recorre el código fuente buscando coincidencias con la expresión regular    
        for mo in re.finditer(self.regex, code, re.MULTILINE):
            start = mo.start() # Posición de inicio cuando se encuentra una coincidencia
            current_line_start = line_start # Posición de inicio de la línea actual
            
            if '\n' in code[current_line_start:start]: # Si se detecta un salto de linea
                invalid = code[last_pos:start] # Se obtiene el segmento inválido
                line_num += invalid.count('\n') # Se cuenta el número de saltos de línea
                last_newline = invalid.rfind('\n') + 1 # Se obtiene la posición del último salto de línea
                line_start = start - (len(invalid) - last_newline) # Se actualiza la posición de inicio de la línea
                
            if start > last_pos: # Si hay un segmento inválido entre la última coincidencia y la actual
                invalid = code[last_pos:start] # Se obtiene el segmento inválido
                self.add_error(invalid, line_num, (last_pos - line_start) + 1) # Se agrega el error a la lista de errores
                
            kind = mo.lastgroup # Se obtiene el tipo de token
            value = mo.group() # Se obtiene el valor del token
            col = start - line_start + 1 # Se obtiene la posición de la columna
            color = dict((name, color) for name, _, color in self.TOKEN_SPEC).get(kind) # Se obtiene el color del token
            
            if kind in ['ERROR', 'NUM_REAL_INCOMPLETO']: # Si el token es un error o un número real incompleto
                self.add_error(value, line_num, col) # Se agrega el error a la lista de errores
            else: # Si el token es válido
                if kind not in ['COMENTARIO_MULTI', 'COMENTARIO_LINEA']: # y no es un comentario
                    self.tokens.append((value, kind, line_num, col, color)) # Se agrega el token a la lista de tokens
            
            last_pos = mo.end() # Se actualiza la posición de la última coincidencia
            
            if '\n' in value: # Si el token contiene un salto de línea
                line_num += value.count('\n') # Se cuenta el número de saltos de línea
                line_start = mo.end() - (len(value) - value.rfind('\n') - 1) # Se actualiza la posición de inicio de la línea
        
        if last_pos < len(code): # Si hay un segmento inválido al final del código
            invalid = code[last_pos:] # Se obtiene el segmento inválido
            self.add_error(invalid, line_num, last_pos - line_start + 1) # Se agrega el error a la lista de errores

        return self.tokens, self.errors # Retorna los tokens y errores encontrados
    
    # Método para agregar errores a la lista de errores
    def add_error(self, text, line, initial_col):
        current_line = line # Línea actual
        current_col = initial_col # Columna inicial
        start_col = current_col  # Inicio del segmento inválido
        segment = [] # Segmento inválido
    
        for char in text: # Se recorre el segmento inválido
            if char == '\n': 
                # Finalizar segmento en esta línea
                if segment: 
                    self.errors.append({
                        'value': ''.join(segment),
                        'line': current_line,
                        'col': start_col,
                        'start': start_col - 1,
                        'length': len(segment)
                    }) # Se agrega el error a la lista de errores
                    segment = [] # Se reinicia el segmento
                current_line += 1 # Se incrementa el número de línea
                current_col = 1 # Se reinicia la columna
                start_col = current_col # Se reinicia la columna inicial
            else: # Si el carácter no es un salto de línea
                if not char.isspace(): # Si no es un espacio en blanco
                    if not segment: # Si el segmento está vacío
                        start_col = current_col # Se actualiza la columna inicial
                    segment.append(char) # Se agrega el carácter al segmento
                else: # Si es un espacio en blanco
                    if segment: # Si el segmento no está vacío
                        self.errors.append({ 
                            'value': ''.join(segment),
                            'line': current_line,
                            'col': start_col,
                            'start': start_col - 1,
                            'length': len(segment)
                        }) # Se agrega el error a la lista de errores
                        segment = [] # Se reinicia el segmento
                current_col += 1 # Se incrementa la columna actual
        if segment: # Si hay un segmento inválido al final
            self.errors.append({ 
                'value': ''.join(segment),
                'line': current_line,
                'col': start_col,
                'start': start_col - 1,
                'length': len(segment)
                }) # Se agrega el error a la lista de errores
    
# Definición de la clase SyntaxHighlighter
# Esta clase se encarga de resaltar la sintaxis del código fuente
class SyntaxHighlighter(QSyntaxHighlighter):
    
    # Inicializando el resaltador de sintaxis
    def __init__(self, document):
        super().__init__(document)
        self.rules = [] # Lista para almacenar las reglas de resaltado
        self.comment_start = QRegularExpression(r'/\*') # Patrón de inicio de comentario multi-línea
        self.comment_end = QRegularExpression(r'\*/') # Patrón de fin de comentario multi-línea
        self.comment_format = QTextCharFormat() # Formato para comentarios
        self.error_positions = {} # Diccionario para almacenar posiciones de errores
        self.error_format = QTextCharFormat() # Formato para errores
        self.error_format.setUnderlineColor(QColor('#FF0000')) # Color de subrayado para errores
        self.error_format.setUnderlineStyle(QTextCharFormat.WaveUnderline) # Estilo de subrayado para errores
        self.string_format = QTextCharFormat() # Formato para cadenas de texto
        self.string_format.setForeground(QColor('#CE9178')) # Color para cadenas de texto
        
        self.double_quote_start = QRegularExpression(r'"') # Patrón de inicio de cadena de texto con comillas dobles
        self.single_quote_start = QRegularExpression(r"'") # Patrón de inicio de cadena de texto con comillas simples
        
        self.quote_end = QRegularExpression(r'["\']') # Patrón de fin de cadena de texto
        
        # Construccion de las reglas de resaltado
        token_spec = [
        # Errores
        ('NUM_REAL_INCOMPLETO', r'\d+\.(?!\d)', QColor('#FF0000')),
        ('ERROR', r'.', QColor('#FF0000')),
        # Numeros
        ('NUM_REAL', r'\d+\.\d+', QColor('#1788ff')),
        ('NUM_ENTERO', r'\d+', QColor('#1788ff')),
        # Booleano
        ('BOOLEANO', r'\b(true|false)\b', QColor('#569CD6')),
        # Identificadores
        ('IDENTIFICADOR', r'[a-zA-Z_][a-zA-Z0-9_]*', QColor('#ff56e3')),
        # Palabras reservadas
        ('PALABRA_RESERVADA', r'\b(if|then|else|end|do|until|while|switch|case|int|float|main|cin|cout|bool)\b', QColor('#569CD6')),
        # Operadores y símbolos
        ('OPERADOR_ARIT', r'(\+\+|--|\+|-|\*|/|%|\^)', QColor('#e789ff')),
        ('ASIGNACION', r'=', QColor('#FFD700')),
        ('OPERADOR_LOG', r'(\&\&|\|\||!)', QColor('#23eeff')),
        ('OPERADOR_REL', r'(<=|>=|==|!=|<|>)', QColor('#23eeff')),
        ('OPERADOR_ES', r'(>>|<<)', QColor('#DCDCAA')),
        ('COMA', r',', QColor('#FFD700')),
        ('SIMBOLOS', r'[\(\)\[\]\{\};]', QColor('#FFD700')),
        # Cadenas de texto
        ('CADENA_DOBLE', r'"(?:\\.|[^"\\])*"', QColor('#CE9178')),
        ('CADENA_SIMPLE', r"'(?:\\.|[^'\\])*'", QColor('#CE9178')),
        # Comentarios
        ('COMENTARIO_MULTI', r'/\*[\s\S]*?\*/', QColor('#6A9955')),
        ('COMENTARIO_LINEA', r'//.*', QColor('#6A9955')),
        ]
        # Se agregan las reglas de resaltado a la lista
        for name, pattern, color in token_spec:
            fmt = QTextCharFormat() # Se crea un formato de texto
            fmt.setForeground(color) # Se establece el color del formato
            self.rules.append((
                QRegularExpression(pattern),
                fmt
            )) # Se agrega el patrón y el formato a la lista de reglas
        self.comment_format.setForeground(QColor('#6A9955')) # Color para comentarios
    
    # Método para resaltar el bloque de texto
    def highlightBlock(self, text):
        self.process_strings(text, self.double_quote_start) # Procesar cadenas de texto con comillas dobles
        self.process_strings(text, self.single_quote_start) # Procesar cadenas de texto con comillas simples
        
        # Aplicar reglas normales
        for pattern, fmt in self.rules:
            iterator = pattern.globalMatch(text) # Iterador para buscar coincidencias
            while iterator.hasNext():
                match = iterator.next() # Obtener la coincidencia
                # Verificar si el formato actual es diferente al nuevo
                if self.format(match.capturedStart()).foreground().color() != self.string_format.foreground().color():
                    self.setFormat(match.capturedStart(), match.capturedLength(), fmt) # Aplicar formato
        
        # Manejar comentarios multi-línea
        self.setCurrentBlockState(0) # Reiniciar el estado del bloque
        start = 0 # Posición de inicio
        
        # Si el bloque anterior no es un comentario, buscar el inicio del comentario
        if self.previousBlockState() != 1:
            start_match = self.comment_start.match(text) # Buscar el inicio del comentario
            start = start_match.capturedStart() # Obtener la posición de inicio del comentario
        
        # Procesar comentarios multi-línea
        while start >= 0:
            end_match = self.comment_end.match(text, start) # Buscar el fin del comentario
            end = end_match.capturedEnd() # Obtener la posición de fin del comentario
            if end == -1: # Si no se encuentra el fin del comentario
                self.setCurrentBlockState(1) # Cambiar el estado del bloque
                comment_length = len(text) - start # Longitud del comentario
            else: # Si se encuentra el fin del comentario
                comment_length = end - start # Longitud del comentario
            
            self.setFormat(start, comment_length, self.comment_format) # Aplicar formato al comentario
            start = self.comment_start.match(text, start + comment_length).capturedStart() # Buscar el siguiente comentario
            
        current_line = self.currentBlock().blockNumber() + 1 # Obtener el número de línea actual
        if current_line in self.error_positions: # Si hay errores en la línea actual
            for start, length in self.error_positions[current_line]: # Obtener las posiciones de los errores
                self.setFormat(start, length, self.error_format) # Aplicar formato de error
        
    # Método para procesar cadenas de texto       
    def process_strings(self, text, pattern):
        self.setCurrentBlockState(0) # Reiniciar el estado del bloque
        start = pattern.match(text).capturedStart() # Buscar el inicio de la cadena
        
        # Procesar cadenas de texto
        while start >= 0:
            end = self.find_string_end(text, start + 1) # Buscar el fin de la cadena
            
            if end == -1: # Si no se encuentra el fin de la cadena
                self.setCurrentBlockState(1) # Cambiar el estado del bloque
                length = len(text) - start # Longitud de la cadena
            else: # Si se encuentra el fin de la cadena
                length = end - start + 1 # Longitud de la cadena
                
            self.setFormat(start, length, self.string_format) # Aplicar formato a la cadena
            start = pattern.match(text, start + length).capturedStart() # Buscar la siguiente cadena

    # Método para encontrar el fin de una cadena de texto
    def find_string_end(self, text, start): # Buscar el fin de la cadena
        escape = False # Variable para manejar caracteres de escape
        
        for i in range(start, len(text)): # Recorrer el texto desde el inicio de la cadena
            if escape: # Si el carácter anterior era un escape
                escape = False # Reiniciar la variable de escape
                continue # Continuar al siguiente carácter
            if text[i] == '\\': # Si se encuentra un carácter de escape
                escape = True # Activar la variable de escape
            elif self.quote_end.match(text[i]).hasMatch(): # Si se encuentra el fin de la cadena
                return i # Retornar la posición del fin de la cadena
        return -1 # Si no se encuentra el fin de la cadena, retornar -1

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.current_index = 0
        self.ast = None
        self.errors = []
        self.stack = []
        self.error_recovery_points = [';', '}', 'end', 'else', 'until']

    def current_token(self):
        return self.tokens[self.current_index] if self.current_index < len(self.tokens) else None

    def advance(self):
        self.current_index += 1

    def match(self, expected_type, expected_lexeme=None, insert_virtual=False):
        token = self.current_token()
        if not token:
            return False
            
        token_type, lexeme, line, col = token
        match_type = token_type == expected_type
        match_lexeme = expected_lexeme is None or lexeme == expected_lexeme
        
        if not (match_type and match_lexeme) and insert_virtual:
            # Insertar token virtual y reportar error
            virtual_token = (expected_type, expected_lexeme, line, col)
            self.errors.append({
                'message': f"Se insertó virtualmente: '{expected_lexeme}'",
                'line': line,
                'col': col
            })
            return True  # Aceptar token virtual
            
        return match_type and match_lexeme

    def error(self, message, token=None):
        if token:
            self.errors.append({
                'message': message,
                'line': token[2],
                'col': token[3]
            })
        else:
            self.errors.append({'message': message})
        
        # Sincronización avanzada
        self.synchronize()

    def parse(self):
        try:
            self.ast = self.programa()
            if self.current_token():
                self.error("Código después del programa principal")
        except Exception as e:
            self.error(f"Error inesperado: {str(e)}")
        return self.ast, self.errors

    def synchronize(self):
        """Saltar tokens hasta encontrar un punto de sincronización"""
        while self.current_token():
            # Puntos de recuperación extendidos
            if self.current_token()[0] in [';', '}'] or \
                self.current_token()[1] in ['if', 'while', 'do', 'cin', 'cout', 'int', 'float', 'end', 'else', 'until'] or \
                self.current_token()[0] == 'IDENTIFICADOR':  # Inicio de asignaciones
                return
            self.advance()

    def programa(self):
        """programa → main { lista_declaracion }"""
        # Crear nodo raíz del programa
        node = ASTNode("Programa")
        
        # main con recuperación
        if not self.match('PALABRA_RESERVADA', 'main'):
            self.error("Se esperaba 'main'", self.current_token())
        else:
            main_token = self.current_token()
            main_node = ASTNode("Main", "main", main_token[2], main_token[3])
            node.add_child(main_node)
            self.advance()
        
        # { con inserción virtual si falta
        if not self.match('SIMBOLOS', '{', insert_virtual=True):
            self.error("Se esperaba '{' después de main", self.current_token())
        else:
            self.advance()
        
        decl_list = self.bloque_codigo()
        node.add_child(decl_list)
        
        # } con inserción virtual si falta
        if not self.match('SIMBOLOS', '}', insert_virtual=True):
            self.error("Se esperaba '}' al final del programa", self.current_token())
        else:
            self.advance()
        
        return node
    
    def bloque_codigo(self):
        node = ASTNode("Bloque")
    
        while self.current_token() and not self.match('SIMBOLOS', '}'):
            try:
                decl = self.declaracion()
                if decl:
                    node.add_child(decl)
                else:
                    # Sincronización avanzada en lugar de solo avanzar
                    self.synchronize()
                    if self.current_token():
                        self.advance()
            except Exception as e:
                self.error(f"Error en declaración: {str(e)}")
                self.synchronize()  # Sincronizar después del error
                if self.current_token():
                    self.advance()
                
        return node

    def lista_declaracion(self):
        """lista_declaracion → declaracion lista_declaracion | declaracion"""
        node = ASTNode("LISTA_DECLARACION")
        
        while self.current_token() and not self.match('SIMBOLOS', '}'):
            decl = self.declaracion()
            if decl:
                node.add_child(decl)
            else:
                self.synchronize()
                if self.current_token():
                    self.advance()
        
        return node

    def declaracion(self):
        """declaracion → declaracion_variable | sentencia"""
        token = self.current_token()
        if not token:
            return None
            
        if token[1] in ['int', 'float', 'bool']:
            return self.declaracion_variable()
        else:
            return self.sentencia()

    def declaracion_variable(self):
        """declaracion_variable → tipo lista_identificadores {, lista_identificadores} ;"""
        node = ASTNode("DeclaracionVariable")
        
        # Tipo con valor por defecto si hay error
        tipo_node = self.tipo() or ASTNode("TIPO", "int", 0, 0)
        node.add_child(tipo_node)
        
        # Identificadores con omisión de errores
        found_id = False
        while True:
            if self.match('IDENTIFICADOR'):
                token = self.current_token()
                node.add_child(ASTNode("Variable", token[1], token[2], token[3]))
                self.advance()
                found_id = True
            elif self.match('COMA'):
                self.advance()
                continue
            else:
                if not found_id:
                    self.error("Se esperaba identificador", self.current_token())
                break
        
        # ; con inserción virtual
        if not self.match('SIMBOLOS', ';', insert_virtual=True):
            self.error("Se esperaba ';'", self.current_token())
        else:
            self.advance()
        
        return node

    def tipo(self):
        """tipo → int | float | bool"""
        try:
            token = self.current_token()
            if token and token[1] in ['int', 'float', 'bool']:
                node = ASTNode("TIPO", token[1], token[2], token[3])
                self.advance()
                return node
            self.error("Tipo inválido", token)
            return ASTNode("TIPO", "int", token[2], token[3])  # Valor por defecto
        except:
            return ASTNode("TIPO", "int", 0, 0)

    # Implementación básica de sentencia
    def sentencia(self):
        """sentencia → seleccion | iteracion | repeticion | sent_in | sent_out | asignacion | incremento_sentencia"""
        token = self.current_token()
        if not token:
            return None
            
        if token[1] == 'if':
            return self.seleccion()
        elif token[1] == 'while':
            return self.iteracion()
        elif token[1] == 'do':
            return self.repeticion()
        elif token[1] == 'cin':
            return self.sent_in()
        elif token[1] == 'cout':
            return self.sent_out()
        elif self.match('IDENTIFICADOR'):
            # Verificar si es un incremento/decremento
            next_token = self.tokens[self.current_index+1] if self.current_index+1 < len(self.tokens) else None
            if next_token and next_token[0] == 'OPERADOR_ARIT' and next_token[1] in ['++', '--']:
                return self.incremento_sentencia()
            else:
                node = self.asignacion()  # Obtener nodo de asignación
                # Agregar ; solo después de la asignación completa
                if not self.match('SIMBOLOS', ';'):
                    self.error("Se esperaba ';'", self.current_token())
                else:
                    self.advance()
                return node
        else:
            self.error("Sentencia inválida", token)
            return None
        
    def incremento_sentencia(self):
        """incremento_sentencia → id OPERADOR_ARIT ('++' | '--') ;"""
        id_token = self.current_token()
        self.advance()
        op_token = self.current_token()
        self.advance()

        # Create the assignment node
        assign_node = ASTNode("Asignacion", line=id_token[2], col=id_token[3])

        # Left side of assignment
        id_node = ASTNode("ID", value=id_token[1], line=id_token[2], col=id_token[3])
        assign_node.add_child(id_node)

        # Right side of assignment
        op_char = "+" if op_token[1] == "++" else "-"
        expr_node = ASTNode("EXPRESION_BINARIA", value=op_char, line=op_token[2], col=op_token[3])
        expr_node.add_child(ASTNode("Variable", value=id_token[1], line=id_token[2], col=id_token[3]))
        expr_node.add_child(ASTNode("Literal", value="1", line=op_token[2], col=op_token[3]))
        assign_node.add_child(expr_node)

        if not self.match('SIMBOLOS', ';'):
            self.error("Se esperaba ';'", self.current_token())
        else:
            self.advance()

        return assign_node
    
    # Implementaciones básicas de las estructuras (para evitar errores)
    def seleccion(self):
        """seleccion → if expression then lista_sentencias [ else lista_sentencias ] end"""
        if_token = self.current_token()
        node = ASTNode("If", line=if_token[2], col=if_token[3]) if if_token else ASTNode("If", error=True)
        if if_token:
            self.advance()
        else:
            self.error("Se esperaba 'if'")
            return node
        
        # Condición
        expr_node = self.expression() or ASTNode("Expression", error=True)
        node.add_child(expr_node)
        
        # then
        if not self.match('PALABRA_RESERVADA', 'then'):
            self.error("Se esperaba 'then'", self.current_token())
        self.advance()
        
        # Bloque then
        then_block = self.lista_sentencias()
        node.add_child(then_block)
        
        # else (opcional)
        if self.match('PALABRA_RESERVADA', 'else'):
            else_token = self.current_token()  # <--- CAPTURAR TOKEN ANTES DE AVANZAR
            self.advance()
            else_block = self.lista_sentencias() or ASTNode("BloqueElse", error=True)
            else_node = ASTNode("Else", line=else_token[2], col=else_token[3])
            else_node.add_child(else_block)
            node.add_child(else_node)
        
        # end
        if not self.match('PALABRA_RESERVADA', 'end'):
            self.error("Se esperaba 'end' para cerrar if", self.current_token())
        self.advance()
        
        return node

    def iteracion(self):
        """iteracion → while expression lista_sentencias end"""
        while_token = self.current_token()
        node = ASTNode("While", line=while_token[2], col=while_token[3]) if while_token else ASTNode("While", error=True)
        if while_token:
            self.advance()
        else:
            self.error("Se esperaba 'while'")
            return node
        
        # Condición
        expr_node = self.expression() or ASTNode("Expression", error=True)
        node.add_child(expr_node)
        
        # Cuerpo del while
        body_node = self.lista_sentencias() or ASTNode("CuerpoWhile", error=True)
        node.add_child(body_node)
        
        # end
        if not self.match('PALABRA_RESERVADA', 'end'):
            self.error("Se esperaba 'end' para cerrar while", self.current_token())
        self.advance()
        
        return node

    def repeticion(self):
        """repeticion → do lista_sentencias (until | while) expression"""
        do_token = self.current_token()
        node = ASTNode("DoWhile", line=do_token[2], col=do_token[3]) if do_token else ASTNode("DoWhile", error=True)
        if do_token:
            self.advance()
        else:
            self.error("Se esperaba 'do'")
            return node
        
        # Cuerpo del do-while
        body_node = self.lista_sentencias() or ASTNode("CuerpoDoWhile", error=True)
        node.add_child(body_node)
        
        # until o while
        if self.match('PALABRA_RESERVADA', 'until') or self.match('PALABRA_RESERVADA', 'while'):
            keyword = self.current_token()[1]
            node.value = keyword
            self.advance()
        else:
            self.error("Se esperaba 'until' o 'while'", self.current_token())
            node.value = "until"  # Valor por defecto para continuar análisis
        
        # Condición
        expr_node = self.expression() or ASTNode("Expression", error=True)
        node.add_child(expr_node)
        
        return node

    def sent_in(self):
        """sent_in → cin >> id ;"""
        node = ASTNode("ENTRADA")
        
        # cin
        if not self.match('PALABRA_RESERVADA', 'cin'):
            self.error("Se esperaba 'cin'", self.current_token())
        else:
            node.add_child(ASTNode("CIN", "cin", *self.current_token()[2:4]))
            self.advance()
        
        # >>
        if not self.match('OPERADOR_ES', '>>'):
            self.error("Se esperaba '>>'", self.current_token())
        else:
            node.add_child(ASTNode("OPERADOR_ENTRADA", ">>", *self.current_token()[2:4]))
            self.advance()
        
        # id
        if not self.match('IDENTIFICADOR'):
            self.error("Se esperaba identificador", self.current_token())
            # Insertar identificador ficticio
            id_node = ASTNode("ERROR ID", "var_temporal", self.current_token()[2] if self.current_token() else 0, 
                              self.current_token()[3] if self.current_token() else 0, error=True)
            node.add_child(id_node)
        else:
            node.add_child(ASTNode("ID", self.current_token()[1], *self.current_token()[2:4]))
            self.advance()
        
        # ;
        if not self.match('SIMBOLOS', ';'):
            self.error("Se esperaba ';'", self.current_token())
        else:
            self.advance()
            
        return node

    def sent_out(self):
        """sent_out → cout << salida ;"""
        node = ASTNode("SALIDA")
        
        # cout
        if not self.match('PALABRA_RESERVADA', 'cout'):
            self.error("Se esperaba 'cout'", self.current_token())
        else:
            node.add_child(ASTNode("COUT", "cout", *self.current_token()[2:4]))
            self.advance()
        
        # <<
        if not self.match('OPERADOR_ES', '<<'):
            self.error("Se esperaba '<<'", self.current_token())
        else:
            node.add_child(ASTNode("OPERADOR_SALIDA", "<<", *self.current_token()[2:4]))
            self.advance()
        
        # salida
        salida_node = self.salida() or ASTNode("Salida", error=True)
        node.add_child(salida_node)
        
        if not self.match('SIMBOLOS', ';'):
            self.error("Se esperaba ';'", self.current_token())
        else:
            self.advance()
        
        return node
    
    def salida(self):
        """salida → cadena | expresion | cadena << expresion | expresion << cadena"""
        node = ASTNode("SALIDA_EXPR")
        
        # Cadena o expresión
        if self.match('CADENA_DOBLE') or self.match('CADENA_SIMPLE'):
            node.add_child(ASTNode("CADENA", self.current_token()[1], *self.current_token()[2:4]))
            self.advance()
        else:
            expr_node = self.expression()
            if expr_node:
                node.add_child(expr_node)
            else:
                self.error("Se esperaba cadena o expresión", self.current_token())
                # Insertar cadena ficticia
                node.add_child(ASTNode("CADENA", "\"texto_temporal\"", 
                                      self.current_token()[2] if self.current_token() else 0, 
                                      self.current_token()[3] if self.current_token() else 0, 
                                      error=True))
        return node

    def asignacion(self):
        """asignacion → id = sent_expression"""
        node = ASTNode("Asignacion")

        # id
        if not self.match('IDENTIFICADOR'):
            self.error("Se esperaba identificador", self.current_token())
            id_node = ASTNode("ID", "var_temporal", self.current_token()[2], self.current_token()[3], error=True)
            self.advance()
        else:
            id_node = ASTNode("ID", self.current_token()[1], *self.current_token()[2:4])
            self.advance()
        node.add_child(id_node)

        # =
        if not self.match('ASIGNACION', '='):
            self.error("Se esperaba '='", self.current_token())
        else:
            self.advance()

        # expresion_asignacion (puede ser otra asignación o expresión)
        expr_node = self.expresion_asignacion()
        node.add_child(expr_node)
    
        return node  # Eliminar consumo de ; aquí
    
    def expresion_asignacion(self):
        """expresion_asignacion → asignacion | expression"""
        # Verificar si es una asignación anidada (ej: a = b = 5)
        if self.match('IDENTIFICADOR'):
            next_token = self.tokens[self.current_index+1] if self.current_index+1 < len(self.tokens) else None
            if next_token and next_token[0] == 'ASIGNACION' and next_token[1] == '=':
                # Parsear como asignación múltiple
                return self.asignacion()
    
        # Si no es asignación, parsear como expresión normal
        return self.expression()

    def sent_expression(self):
        """sent_expression → expresion_asignacion ; | ;"""
        node = ASTNode("SENT_EXPRESSION")
    
        if not self.match('SIMBOLOS', ';'):
            expr_node = self.expresion_asignacion()
            if expr_node:
                node.add_child(expr_node)
    
        if not self.match('SIMBOLOS', ';'):
            self.error("Se esperaba ';'", self.current_token())
        else:
            self.advance()
        
        return node

    def lista_sentencias(self):
        """lista_sentencias → lista_sentencias sentencia | ε"""
        node = ASTNode("LISTA_SENTENCIAS")
        stack = 0
        
        while self.current_token() and not self.match('SIMBOLOS', '}'):
            token = self.current_token()
            token_type, lexeme, line, col = token
            
            # Manejar inicio de bloques
            if lexeme in ['if', 'while', 'do']:
                stack += 1
            # Manejar fin de bloques
            elif lexeme == 'end':
                if stack > 0:
                    stack -= 1
                else:
                    break
            
            # Detenerse solo en palabras clave de cierre en nivel superior
            if stack == 0 and lexeme in ['end', 'else', 'until', 'while', '}']:
                break
            
            # Parsear sentencia
            try:
                sent = self.sentencia()
                if sent:
                    node.add_child(sent)
                else:
                    break
            except Exception as e:
                self.error(f"Error procesando sentencia: {str(e)}")
                # Insertar sentencia ficticia
                node.add_child(ASTNode("SentenciaError", "error", line, col, error=True))
                self.synchronize()
                if self.current_token():
                    self.advance()
        
        return node
    
    def expression(self):
        """expression → expression_term [ log_op expression ]"""
        left = self.expression_term()
        if not left:
            # Crear nodo de error pero continuar
            left = ASTNode("ExpressionTerm", error=True)
            self.error("Expresión inválida", self.current_token())
        
        node = left
        
        while self.current_token() and self.current_token()[1] in ['&&', '||']:
            op_token = self.current_token()
            op_node = ASTNode("OperacionLogica", op_token[1], op_token[2], op_token[3])
            self.advance()
            
            right = self.expression_term()
            if not right:
                # Crear nodo de error pero continuar
                right = ASTNode("ExpressionTerm", error=True)
                self.error("Falta operando derecho", self.current_token())
            
            op_node.add_child(node)
            op_node.add_child(right)
            node = op_node
            
        return node
    
    def expression_term(self):
        """expression_term → expression_simple [ rel_op expression_simple ]"""
        left = self.expression_simple()
        if not left:
            # Crear nodo de error pero continuar
            left = ASTNode("ExpressionSimple", error=True)
            self.error("Término de expresión inválido", self.current_token())
        
        node = left
        
        while self.current_token() and self.current_token()[1] in ['<', '<=', '>', '>=', '==', '!=']:
            op_token = self.current_token()
            self.advance()
            right = self.expression_simple()
            if not right:
                # Crear nodo de error pero continuar
                right = ASTNode("ExpressionSimple", error=True)
                self.error("Falta operando derecho en operación relacional", self.current_token())
            
            op_node = ASTNode("EXPRESION_BINARIA", op_token[1], op_token[2], op_token[3])
            op_node.add_child(node)
            op_node.add_child(right)
            node = op_node
            
        return node

    def expression_simple(self):
        """expression_simple → expression_simple suma_op termino | termino"""
        left = self.termino()
        if not left:
            # Crear nodo de error pero continuar
            left = ASTNode("Termino", error=True)
            self.error("Término simple inválido", self.current_token())
        
        node = left
        
        while self.current_token() and self.current_token()[1] in ['+', '-']:
            op_token = self.current_token()
            self.advance()
            right = self.termino()
            if not right:
                # Crear nodo de error pero continuar
                right = ASTNode("Termino", error=True)
                self.error("Falta operando derecho en suma/resta", self.current_token())
            
            op_node = ASTNode("EXPRESION_BINARIA", op_token[1], op_token[2], op_token[3])
            op_node.add_child(node)
            op_node.add_child(right)
            node = op_node
            
        return node

    def termino(self):
        """termino → termino mult_op factor | factor"""
        left = self.factor()
        if not left:
            # Crear nodo de error pero continuar
            left = ASTNode("Factor", error=True)
            self.error("Factor inválido", self.current_token())
        
        node = left
        
        while self.current_token() and self.current_token()[1] in ['*', '/', '%']:
            op_token = self.current_token()
            self.advance()
            right = self.factor()
            if not right:
                # Crear nodo de error pero continuar
                right = ASTNode("Factor", error=True)
                self.error("Falta operando derecho en multiplicación/división", self.current_token())
            
            op_node = ASTNode("EXPRESION_BINARIA", op_token[1], op_token[2], op_token[3])
            op_node.add_child(node)
            op_node.add_child(right)
            node = op_node
            
        return node

    def factor(self):
        """factor → componente | factor pot_op componente"""
        left = self.componente()
        if not left:
            # Crear nodo de error pero continuar
            left = ASTNode("Componente", error=True)
            self.error("Componente inválido", self.current_token())
        
        node = left
        
        while self.current_token() and self.current_token()[1] == '^':
            op_token = self.current_token()
            op_node = ASTNode("Operacion", op_token[1], op_token[2], op_token[3])
            self.advance()
            right = self.componente()
            if not right:
                # Crear nodo de error pero continuar
                right = ASTNode("Componente", error=True)
                self.error("Falta operando derecho en potencia", self.current_token())
            
            op_node.add_child(node)
            op_node.add_child(right)
            node = op_node
            
        return node

    def componente(self):
        """componente → ( expression ) | número | id | bool | op_logico componente | id OPERADOR_ARIT ('++' | '--')"""
        token = self.current_token()
        if not token:
            return ASTNode("Componente", error=True)
        
        # ( expression )
        if self.match('SIMBOLOS', '('):
            self.advance()
            expr_node = self.expression() or ASTNode("Expression", error=True)
            
            if not self.match('SIMBOLOS', ')'):
                self.error("Se esperaba ')'", self.current_token())
            else:
                self.advance()
                
            return expr_node
        
        # Números
        elif self.match('NUM_ENTERO') or self.match('NUM_REAL'):
            node = ASTNode("Literal", token[1], token[2], token[3])
            self.advance()
            return node
        
        # Identificadores
        elif self.match('IDENTIFICADOR'):
            node = ASTNode("Variable", token[1], token[2], token[3])
            self.advance()
            
            # Operador postfijo (++ o --)
            if self.current_token() and self.current_token()[1] in ['++', '--']:
                op_token = self.current_token()
                op_node = ASTNode("OperacionUnaria", op_token[1], op_token[2], op_token[3])
                op_node.add_child(node)
                self.advance()
                return op_node
            
            return node
        
        # Booleanos
        elif self.match('BOOLEANO', 'true') or self.match('BOOLEANO', 'false'):
            node = ASTNode("Literal", token[1], token[2], token[3])
            self.advance()
            return node
        
        # Operadores lógicos unarios
        elif self.match('OPERADOR_LOG', '!'):
            op_token = self.current_token()
            self.advance()
            comp_node = self.componente() or ASTNode("Componente", error=True)
            
            op_node = ASTNode("OperacionUnaria", op_token[1], op_token[2], op_token[3])
            op_node.add_child(comp_node)
            return op_node
        
        else:
            self.error("Componente inválido", token)
            # Crear componente ficticio
            return ASTNode("Error Componente", "0", token[2], token[3], error=True)
        
class SymbolTable:
    def __init__(self):
        self.symbols = {}
        self.ambito_actual = "global"
        self.direccion_memoria = 1
        self.desplazamiento = 1
        
    def insertar(self, nombre, tipo, linea, columna, ambito=None):
        if ambito is None:
            ambito = self.ambito_actual
            
        clave = f"{ambito}::{nombre}"
        if clave in self.symbols:
            return False  # Variable ya declarada
            
        # Asegurarse de que la línea no sea None
        linea_valida = linea if linea is not None else 0
            
        self.symbols[clave] = {
            'tipo': tipo,
            'ambito': ambito,
            'lineas': [linea_valida],  # Iniciar con la línea de declaración
            'columna': columna,
            'direccion': self.direccion_memoria,
            'desplazamiento': self.desplazamiento,
            'apariciones': [],  # NUEVO: Lista para todas las apariciones (línea, columna)
            'valor': None # NUEVO: Para almacenar el valor calculado
        }
        
        # Registrar la declaración como primera aparición
        self.registrar_aparicion(nombre, linea, columna, ambito)
        
        self.direccion_memoria += 1
        self.desplazamiento += 1
        return True
        
    def buscar(self, nombre, ambito=None):
        if ambito is None:
            ambito = self.ambito_actual
            
        # Buscar en ámbito actual primero
        clave = f"{ambito}::{nombre}"
        if clave in self.symbols:
            return self.symbols[clave]
            
        # Buscar en ámbito global
        clave_global = f"global::{nombre}"
        if clave_global in self.symbols:
            return self.symbols[clave_global]
            
        return None
        
    def registrar_uso(self, nombre, linea, columna=None, ambito=None):
        """Registra un nuevo uso de la variable, incluyendo múltiples veces en una misma línea."""
        if linea is None or linea == 0:
            return False

        simbolo = self.buscar(nombre, ambito)
        if simbolo:
            # Registrar aparición exacta
            self.registrar_aparicion(nombre, linea, columna, ambito)

            # También mantener el registro general de líneas
            if linea not in simbolo['lineas']:
                simbolo['lineas'].append(linea)
                simbolo['lineas'] = sorted([l for l in simbolo['lineas'] if l is not None and l > 0])
            return True
        return False

    def registrar_aparicion(self, nombre, linea, columna, ambito=None):
        """NUEVO: Registra cada aparición individual de la variable con línea y columna"""
        if linea is None or linea == 0:
            return False
            
        simbolo = self.buscar(nombre, ambito)
        if simbolo:
            # Siempre agregar la aparición, incluso si está en la misma línea
            aparicion = {'linea': linea, 'columna': columna}
            if aparicion not in simbolo['apariciones']:
                simbolo['apariciones'].append(aparicion)
                # Ordenar apariciones por línea y columna
                simbolo['apariciones'].sort(key=lambda x: (x['linea'], x['columna']))
            return True
        return False
        
    def obtener_lineas_con_apariciones(self, nombre, ambito=None):
        """NUEVO: Obtiene las líneas con el formato correcto para mostrar"""
        simbolo = self.buscar(nombre, ambito)
        if not simbolo:
            return ""
        
        # Crear lista de líneas repetidas según las apariciones
        lineas_expandidas = []
        for aparicion in simbolo['apariciones']:
            lineas_expandidas.append(str(aparicion['linea']))
                
        return ", ".join(lineas_expandidas)
        
    def entrar_ambito(self, nombre_ambito):
        self.ambito_actual = nombre_ambito
        
    def salir_ambito(self):
        self.ambito_actual = "global"
        
    def __str__(self):
        result = "TABLA DE SÍMBOLOS:\n"
        # MODIFICADO: Eliminar "Ámbito", "Apariciones" y "Desplazamiento"
        result += "Nombre\tTipo\tLíneas\tDirección\n"
        result += "-" * 50 + "\n"
        for clave, info in self.symbols.items():
            nombre = clave.split('::')[1]
            # Usar el nuevo método para obtener líneas con conteo de apariciones
            lineas_str = self.obtener_lineas_con_apariciones(nombre, info['ambito'])
            # MODIFICADO: Eliminar info['ambito'], total_apariciones, y info['desplazamiento']
            result += f"{nombre}\t{info['tipo']}\t{lineas_str}\t{info['direccion']}\n"
        return result
    
class SemanticAnalyzer:
    def __init__(self):
        self.tabla_simbolos = SymbolTable()
        self.errores = []
        self.ambito_actual = "global"
        self.ast_anotado = None
        
    def analizar(self, ast):
        self.ast_anotado = ast
        self.visitar_nodo(ast)
        return self.errores
        
    def visitar_nodo(self, nodo):
        if nodo is None:
            return "error"
            
        metodo_name = f"visitar_{nodo.node_type}"
        if hasattr(self, metodo_name):
            return getattr(self, metodo_name)(nodo)
        else:
            # Visitar hijos por defecto
            for hijo in nodo.children:
                self.visitar_nodo(hijo)
            return "void"
    
    def visitar_Programa(self, nodo):
        self.tabla_simbolos.entrar_ambito("global")
        for hijo in nodo.children:
            self.visitar_nodo(hijo)
        self.tabla_simbolos.salir_ambito()
        return "void"
        
    def visitar_DeclaracionVariable(self, nodo):
        if len(nodo.children) < 2:
            self.agregar_error("Declaración de variable incompleta", nodo.line, nodo.col)
            return "error"
            
        tipo_nodo = nodo.children[0]
        tipo = tipo_nodo.value if tipo_nodo else "desconocido"
        
        for i in range(1, len(nodo.children)):
            var_nodo = nodo.children[i]
            if var_nodo.node_type == "Variable":
                nombre = var_nodo.value
                if not self.tabla_simbolos.insertar(nombre, tipo, var_nodo.line, var_nodo.col):
                    self.agregar_error(f"Variable '{nombre}' ya declarada", var_nodo.line, var_nodo.col)
                else:
                    if tipo in ['int', 'float']:
                        simbolo = self.tabla_simbolos.buscar(nombre)
                        simbolo['valor'] = 0
                    
        return "void"
        
    def visitar_Asignacion(self, nodo):
        if len(nodo.children) < 2:
            self.agregar_error("Asignación incompleta", nodo.line, nodo.col)
            return "error"
        
        # Verificar variable izquierda
        id_nodo = nodo.children[0]
        if id_nodo.node_type != "ID":
            self.agregar_error("Se esperaba identificador en asignación", id_nodo.line, id_nodo.col)
            return "error"
        
        nombre_var = id_nodo.value
        simbolo = self.tabla_simbolos.buscar(nombre_var)

        if not simbolo:
            # ERROR: Variable no declarada
            self.agregar_error(f"Variable '{nombre_var}' no declarada", id_nodo.line, id_nodo.col)
            return "error"

        # Registrar el uso de la variable en el lado izquierdo de la asignación
        self.tabla_simbolos.registrar_uso(nombre_var, id_nodo.line, id_nodo.col)

        tipo_var = simbolo['tipo']

        # Verificar expresión derecha
        expr_nodo = nodo.children[1]
        tipo_expr = self.visitar_nodo(expr_nodo)

        # Special case: assignment of a float expression to an int variable
        if tipo_var == 'int' and tipo_expr == 'float' and expr_nodo.node_type != 'Literal':
            if expr_nodo.calculated_value is not None:
                simbolo['valor'] = int(expr_nodo.calculated_value)
                nodo.calculated_value = int(expr_nodo.calculated_value)
        else:
            # General case: check for type compatibility
            if tipo_expr != "error" and not self.tipos_compatibles(tipo_var, tipo_expr):
                self.agregar_error(f"Tipos incompatibles: no se puede asignar {tipo_expr} a {tipo_var}", 
                                id_nodo.line, id_nodo.col)
                nodo.is_error = True
                nodo.calculated_value = "error"
            else:
                # Update symbol table value
                if expr_nodo.calculated_value is not None:
                    simbolo['valor'] = expr_nodo.calculated_value
                    nodo.calculated_value = expr_nodo.calculated_value

        # Annotate types in the AST
        nodo.tipo = tipo_var
        id_nodo.tipo = tipo_var
        if hasattr(expr_nodo, 'tipo'):
            expr_nodo.tipo = tipo_expr
        
        return tipo_var

    def visitar_ENTRADA(self, nodo):
        """Manejar cin >> variable"""
        if len(nodo.children) >= 3:
            id_nodo = nodo.children[2]  # El tercer hijo es el ID
            if id_nodo and id_nodo.node_type == "ID":
                nombre_var = id_nodo.value
                simbolo = self.tabla_simbolos.buscar(nombre_var)
        
                if not simbolo:
                    # ERROR: Variable no declarada
                    linea = id_nodo.line if id_nodo.line is not None else 0
                    columna = id_nodo.col if id_nodo.col is not None else 0
                    self.agregar_error(f"Variable '{nombre_var}' no declarada", linea, columna)
                    
                # NUEVO: Registrar aparición de la variable en entrada
                self.tabla_simbolos.registrar_aparicion(nombre_var, nodo.line, nodo.col)

        return "void"

    def visitar_SALIDA(self, nodo):
        """Manejar cout << expresión"""
        # cout
        if len(nodo.children) > 0:
            self.visitar_nodo(nodo.children[0])
    
        # <<
        if len(nodo.children) > 1:
            self.visitar_nodo(nodo.children[1])
    
        # expresión de salida
        if len(nodo.children) > 2:
            expr_nodo = nodo.children[2]
            tipo_expr = self.visitar_nodo(expr_nodo)
        
            # Registrar uso de variables en la expresión de salida
            self._registrar_usos_en_expresion(expr_nodo)
        
        return "void"
    
    def _registrar_usos_en_expresion(self, nodo):
        """Función auxiliar para registrar usos de variables en expresiones"""
        if nodo is None:
            return

        # Registrar si es un nodo de variable/identificador
        if nodo.node_type in ["ID", "Variable"]:
            nombre_var = nodo.value
            simbolo = self.tabla_simbolos.buscar(nombre_var)
            if simbolo:
                linea = nodo.line if nodo.line is not None else 0
                columna = nodo.col if nodo.col is not None else 0
                self.tabla_simbolos.registrar_uso(nombre_var, linea, columna)

        # Recursivamente visitar hijos
        for hijo in nodo.children:
            self._registrar_usos_en_expresion(hijo)
        
    def visitar_If(self, nodo):
        if len(nodo.children) >= 1:
            cond_nodo = nodo.children[0]
            tipo_cond = self.visitar_nodo(cond_nodo)
            
            # Registrar uso de variables en la condición
            self._registrar_usos_en_expresion(cond_nodo)
            
            if tipo_cond != "bool" and tipo_cond != "error":
                self.agregar_error("La condición del if debe ser booleana", cond_nodo.line, cond_nodo.col)
                
        # Visitar bloques then y else
        for i in range(1, len(nodo.children)):
            self.visitar_nodo(nodo.children[i])
            
        return "void"
    
    def visitar_INCREMENTO_SENTENCIA(self, nodo):
        """Manejar id++ o id--"""
        if len(nodo.children) >= 1:
            id_nodo = nodo.children[0]
            if id_nodo and id_nodo.node_type == "ID":
                nombre_var = id_nodo.value
                simbolo = self.tabla_simbolos.buscar(nombre_var)
            
                if not simbolo:
                    # ERROR: Variable no declarada
                    self.agregar_error(f"Variable '{nombre_var}' no declarada", id_nodo.line, id_nodo.col)
                else:
                    # Verificar que sea numérico
                    if simbolo['tipo'] not in ['int', 'float']:
                        self.agregar_error(f"Incremento no aplicable a tipo {simbolo['tipo']}", 
                                         id_nodo.line, id_nodo.col)
                    # Registrar uso de la variable en esta línea
                    self.tabla_simbolos.registrar_uso(nombre_var, nodo.line)
                    
                # Visitar el nodo ID para registrar el uso
                self.visitar_nodo(id_nodo)
    
        return "void"
    
    def visitar_INCREMENTO(self, nodo):
        """Manejar nodos de incremento específicos"""
        return self.visitar_OperacionUnaria(nodo)
        
    def visitar_While(self, nodo):
        if len(nodo.children) >= 1:
            cond_nodo = nodo.children[0]
            tipo_cond = self.visitar_nodo(cond_nodo)
            
            # Registrar uso de variables en la condición
            self._registrar_usos_en_expresion(cond_nodo)
            
            if tipo_cond != "bool" and tipo_cond != "error":
                self.agregar_error("La condición del while debe ser booleana", cond_nodo.line, cond_nodo.col)
                
        # Visitar cuerpo
        for i in range(1, len(nodo.children)):
            cuerpo_nodo = nodo.children[i]
            self.visitar_nodo(cuerpo_nodo)
            # Registrar usos en el cuerpo del while
            self._registrar_usos_en_expresion(cuerpo_nodo)
            
        return "void"

    def visitar_DoWhile(self, nodo):
        # Visitar cuerpo primero
        if len(nodo.children) >= 1:
            cuerpo_nodo = nodo.children[0]
            self.visitar_nodo(cuerpo_nodo)
            
        # Luego condición
        if len(nodo.children) >= 2:
            cond_nodo = nodo.children[1]
            tipo_cond = self.visitar_nodo(cond_nodo)
            
            # Registrar uso de variables en la condición
            self._registrar_usos_en_expresion(cond_nodo)
            
            if tipo_cond != "bool" and tipo_cond != "error":
                self.agregar_error("La condición del do-while debe ser booleana", cond_nodo.line, cond_nodo.col)
                
        return "void"
        
    def visitar_ID(self, nodo):
        nombre = nodo.value
        simbolo = self.tabla_simbolos.buscar(nombre)
            
        if not simbolo:
            # ERROR: Variable no declarada
            self.agregar_error(f"Variable '{nombre}' no declarada", nodo.line, nodo.col)
            nodo.tipo = "error"
            return "error"
                
        # NUEVO: Registrar cada aparición individual con línea y columna
        self.tabla_simbolos.registrar_aparicion(nombre, nodo.line, nodo.col)
                
        nodo.tipo = simbolo['tipo']
        nodo.calculated_value = simbolo['valor']
        return simbolo['tipo']

    def visitar_Variable(self, nodo):
        nombre = nodo.value
        simbolo = self.tabla_simbolos.buscar(nombre)
            
        if not simbolo:
            # ERROR: Variable no declarada
            self.agregar_error(f"Variable '{nombre}' no declarada", nodo.line, nodo.col)
            nodo.tipo = "error"
            return "error"
                
        # NUEVO: Registrar cada aparición individual con línea y columna
        self.tabla_simbolos.registrar_aparicion(nombre, nodo.line, nodo.col)
                
        nodo.tipo = simbolo['tipo']
        nodo.calculated_value = simbolo['valor']
        return simbolo['tipo']
        
    def visitar_Literal(self, nodo):
        valor = nodo.value
    
        # Determinar tipo del literal
        if valor.isdigit():
            nodo.tipo = "int"
            nodo.calculated_value = int(valor)
            return "int"
        elif self.es_numero_real(valor):
            nodo.tipo = "float" 
            nodo.calculated_value = float(valor)
            return "float"
        elif valor in ['true', 'false']:
            nodo.tipo = "bool"
            nodo.calculated_value = (valor == 'true')
            return "bool"
        elif valor.startswith('"') or valor.startswith("'"):
            nodo.tipo = "string"
            nodo.calculated_value = valor[1:-1]
            return "string"
        else:
            # Intentar convertir a número
            try:
                nodo.calculated_value = float(valor)
                nodo.tipo = "float"
                return "float"
            except:
                nodo.tipo = "desconocido"
                return "desconocido"

    def visitar_OperacionUnaria(self, nodo):
        """Manejar operaciones unarias como ++ y --"""
        if len(nodo.children) == 1:
            operando = nodo.children[0]
            tipo_operando = self.visitar_nodo(operando)

            # Si no se puede determinar el tipo, asumir int
            if tipo_operando == "void" or tipo_operando is None:
                tipo_operando = "int"
            
            if tipo_operando not in ['int', 'float']:
                self.agregar_error(f"Operación unaria '{nodo.value}' no aplicable a tipo {tipo_operando}", 
                               nodo.line, nodo.col)
                return "error"
        
            nodo.tipo = tipo_operando
            return tipo_operando
        return "error"
            
    def visitar_OperacionLogica(self, nodo):
        if len(nodo.children) != 2:
            self.agregar_error("Operación lógica incompleta", nodo.line, nodo.col)
            return "error"

        izquierda = nodo.children[0]
        derecha = nodo.children[1]

        tipo_izq = self.visitar_nodo(izquierda)
        tipo_der = self.visitar_nodo(derecha)

        self._registrar_usos_en_expresion(izquierda)
        self._registrar_usos_en_expresion(derecha)

        if tipo_izq != 'bool' or tipo_der != 'bool':
            self.agregar_error(f"Operador '{nodo.value}' requiere operandos booleanos", nodo.line, nodo.col)
            return "error"

        val_izq = izquierda.calculated_value
        val_der = derecha.calculated_value

        if val_izq is not None and val_der is not None:
            try:
                if nodo.value == '&&':
                    nodo.calculated_value = val_izq and val_der
                elif nodo.value == '||':
                    nodo.calculated_value = val_izq or val_der
            except TypeError:
                self.agregar_error(f"Operación '{nodo.value}' no se puede realizar con los tipos de datos", nodo.line, nodo.col)

        nodo.tipo = 'bool'
        return 'bool'

            
    def visitar_EXPRESION_BINARIA(self, nodo):
        if len(nodo.children) != 2:
            self.agregar_error("Operación binaria incompleta", nodo.line, nodo.col)
            return "error"
        
        izquierda = nodo.children[0]
        derecha = nodo.children[1]
    
        tipo_izq = self.visitar_nodo(izquierda)
        tipo_der = self.visitar_nodo(derecha)
        
        self._registrar_usos_en_expresion(izquierda)
        self._registrar_usos_en_expresion(derecha)
        
        # Si no se pueden determinar los tipos, asumir int
        if tipo_izq == "void" or tipo_izq is None:
            tipo_izq = "int"
        if tipo_der == "void" or tipo_der is None:
            tipo_der = "int"
    
        operador = nodo.value
        
        # Calcular valor
        val_izq = izquierda.calculated_value
        val_der = derecha.calculated_value
        if val_izq is not None and val_der is not None:
            try:
                if operador == '+': nodo.calculated_value = val_izq + val_der
                elif operador == '-': nodo.calculated_value = val_izq - val_der
                elif operador == '*': nodo.calculated_value = val_izq * val_der
                elif operador == '/':
                    if tipo_izq == 'int' and tipo_der == 'int':
                        nodo.calculated_value = int(val_izq // val_der)
                    else:
                        nodo.calculated_value = float(val_izq) / float(val_der)
                elif operador == '%': nodo.calculated_value = val_izq % val_der
                elif operador == '^': nodo.calculated_value = val_izq ** val_der
                elif operador == '<': nodo.calculated_value = val_izq < val_der
                elif operador == '<=': nodo.calculated_value = val_izq <= val_der
                elif operador == '>': nodo.calculated_value = val_izq > val_der
                elif operador == '>=': nodo.calculated_value = val_izq >= val_der
                elif operador == '==': nodo.calculated_value = val_izq == val_der
                elif operador == '!=': nodo.calculated_value = val_izq != val_der
            except TypeError:
                self.agregar_error(f"Operación '{operador}' no se puede realizar con los tipos de datos", nodo.line, nodo.col)

    
        # Verificar tipos según operador
        tipo_resultado = self.verificar_operacion_binaria(operador, tipo_izq, tipo_der, nodo.line, nodo.col)
        nodo.tipo = tipo_resultado
    
        return tipo_resultado
        
    def verificar_operacion_binaria(self, operador, tipo_izq, tipo_der, linea, columna):
        # Si alguno es error, propagar error
        if tipo_izq == "error" or tipo_der == "error":
            return "error"
            
        # Operadores aritméticos
        if operador in ['+', '-', '*', '/', '%', '^']:
            if tipo_izq in ['int', 'float'] and tipo_der in ['int', 'float']:
                if operador == '/':
                    if tipo_izq == 'int' and tipo_der == 'int':
                        return 'int'
                    else:
                        return 'float'
                # Promoción de tipos: si alguno es float, resultado es float
                if tipo_izq == 'float' or tipo_der == 'float':
                    return 'float'
                return 'int'
            else:
                self.agregar_error(f"Operador '{operador}' no aplicable a tipos {tipo_izq} y {tipo_der}", 
                                 linea, columna)
                return "error"
                
        # Operadores relacionales
        elif operador in ['<', '<=', '>', '>=', '==', '!=']:
            if (tipo_izq in ['int', 'float'] and tipo_der in ['int', 'float']) or \
               (tipo_izq == tipo_der and tipo_izq in ['bool', 'string']):
                return 'bool'
            else:
                self.agregar_error(f"Operador '{operador}' no aplicable a tipos {tipo_izq} y {tipo_der}", 
                                 linea, columna)
                return "error"
                
        return "error"
        
    def tipos_compatibles(self, tipo1, tipo2):
        if tipo1 == tipo2:
            return True
        # Conversiones implícitas permitidas
        if tipo1 == 'float' and tipo2 == 'int':
            return True
        # No permitir asignar float a int (pérdida de precisión)
        if tipo1 == 'int' and tipo2 == 'float':
            return False
        return False
        
    def es_numero_real(self, texto):
        try:
            float(texto)
            return True
        except ValueError:
            return False
            
    def agregar_error(self, mensaje, linea, columna):
        self.errores.append({
            'mensaje': mensaje,
            'linea': linea,
            'columna': columna,
            'tipo': 'semantico'
        })

class CodeGenerator:
    def __init__(self):
        self.code = []
        self.var_addresses = {}
        self.temp_counter = 0
        self.label_counter = 0
        self.current_address = 0
        self.next_temp_addr = 100
        self.string_data = []
        self.string_counter = 0
        
    def generate(self, ast_node):
        """Genera código ejecutable a partir del AST anotado"""
        self.code = []
        self.var_addresses = {}
        self.current_address = 0
        self.next_temp_addr = 100
        self.string_data = []
        self.string_counter = 0
        
        # Inicialización estándar del código
        self.emit("LD  6,0(0)")  # Inicializar puntero de pila
        self.emit("ST  0,0(0)")   # Inicialización
        
        # Recopilar direcciones de variables
        self.collect_variable_addresses(ast_node)
        
        # Inicializar todas las variables a 0
        for addr in range(len(self.var_addresses)):
            self.emit(f"LDC  0,0(0)")
            self.emit(f"ST  0,{addr}(5)")
        
        # Generar código principal
        self.visit_program(ast_node)
        
        # Agregar datos de cadenas si existen
        if self.string_data:
            self.emit("")  # Separador
            self.emit("; Datos de cadenas")
            for i, (label, string) in enumerate(self.string_data):
                # Convertir cadena a valores ASCII
                for j, char in enumerate(string):
                    ascii_val = ord(char)
                    self.emit(f"{label}_{j}: DC  {ascii_val}(0)")
                # Terminador de cadena (0)
                self.emit(f"{label}_end: DC  0(0)")
        
        # Finalizar programa
        self.emit("HALT  0,0,0")
        
        # Generar código numerado
        final_code = []
        line_num = 0
        for line in self.code:
            if line and not line.startswith(';'):  # Saltar líneas de comentario
                final_code.append(f"{line_num:3}:{line}")
                line_num += 1
        
        return "\n".join(final_code)
    
    def add_string_data(self, string_value):
        """Agrega una cadena a los datos y devuelve la etiqueta"""
        label = f"S{self.string_counter}"
        self.string_counter += 1
        self.string_data.append((label, string_value))
        return label
    
    def collect_variable_addresses(self, ast_node):
        """Recopila las direcciones de memoria de las variables del AST"""
        if ast_node is None:
            return
            
        if ast_node.node_type == "DeclaracionVariable":
            for i in range(1, len(ast_node.children)):
                var_node = ast_node.children[i]
                if var_node.node_type == "Variable":
                    self.var_addresses[var_node.value] = self.current_address
                    self.current_address += 1
        
        for child in ast_node.children:
            self.collect_variable_addresses(child)
    
    def emit(self, instruction):
        """Agrega una instrucción al código"""
        self.code.append(f"    {instruction}")
    
    def get_var_address(self, var_name):
        """Obtiene la dirección de una variable"""
        return self.var_addresses.get(var_name)
    
    def new_temp_addr(self):
        """Genera una nueva dirección temporal"""
        addr = self.next_temp_addr
        self.next_temp_addr += 1
        return addr
    
    def new_label(self):
        """Genera una nueva etiqueta (dirección futura)"""
        return len(self.code)  # Devuelve la próxima posición en el código
    
    def patch_label(self, label, target_addr):
        """Parchea una instrucción de salto con la dirección correcta"""
        if label < len(self.code):
            # Extraer la instrucción actual
            current = self.code[label]
            # Reemplazar el marcador de etiqueta con la dirección real
            parts = current.split()
            if len(parts) >= 4:
                parts[3] = f"{target_addr}"
                self.code[label] = "    " + " ".join(parts)
    
    def visit_program(self, node):
        """Visita un nodo de programa"""
        for child in node.children:
            if child.node_type == "Bloque":
                self.visit_block(child)
    
    def visit_block(self, node):
        """Visita un bloque de código"""
        for child in node.children:
            self.visit_statement(child)
    
    def visit_statement(self, node):
        """Visita una sentencia"""
        if node.node_type == "DeclaracionVariable":
            self.visit_declaration(node)
        elif node.node_type == "Asignacion":
            self.visit_assignment(node)
        elif node.node_type == "If":
            self.visit_if(node)
        elif node.node_type == "While":
            self.visit_while(node)
        elif node.node_type == "DoWhile":
            self.visit_dowhile(node)
        elif node.node_type == "ENTRADA":
            self.visit_input(node)
        elif node.node_type == "SALIDA":
            self.visit_output(node)
        elif node.node_type == "LISTA_SENTENCIAS":
            self.visit_block(node)
    
    def visit_declaration(self, node):
        """Visita una declaración de variable (ya inicializada)"""
        pass
    
    def visit_assignment(self, node):
        """Visita una asignación: ID = expresión"""
        if len(node.children) < 2:
            return
            
        id_node = node.children[0]
        expr_node = node.children[1]
        
        # Evaluar expresión y poner resultado en registro 0
        self.visit_expression(expr_node, 0)
        
        # Almacenar en variable
        var_name = id_node.value
        addr = self.get_var_address(var_name)
        if addr is not None:
            self.emit(f"ST  0,{addr}(5)")
    
    def visit_expression(self, node, dest_reg):
        """Evalúa una expresión y pone el resultado en el registro dest_reg"""
        if node.node_type == "Literal":
            self.visit_literal(node, dest_reg)
        elif node.node_type in ["Variable", "ID"]:
            self.visit_variable(node, dest_reg)
        elif node.node_type == "CADENA":
            # Para cadenas en expresiones (no en cout), usar una representación numérica
            self.emit(f"LDC  {dest_reg},0(0)")
        elif node.node_type == "EXPRESION_BINARIA":
            self.visit_binary_expression(node, dest_reg)
        elif node.node_type == "OperacionUnaria":
            self.visit_unary_operation(node, dest_reg)
        elif node.node_type == "OperacionLogica":
            self.visit_logical_operation(node, dest_reg)
        else:
            self.emit(f"LDC  {dest_reg},0(0)")
    
    def visit_cadena(self, node, dest_reg):
        """Maneja una cadena (simplificado para el ejemplo)"""
        # Para simplificar, cargamos la cadena como una constante
        # En un implementación real, se necesitaría manejar cadenas en memoria
        str_value = node.value[1:-1] if node.value else ""  # Quitar comillas
        # Usamos un código numérico para representar la cadena (simplificación)
        str_code = hash(str_value) % 1000 if str_value else 0
        self.emit(f"LDC  {dest_reg},{str_code}(0)")
    
    def visit_literal(self, node, dest_reg):
        """Carga un literal en el registro dest_reg"""
        value = node.value
        
        if node.tipo == "bool":
            int_value = 1 if value == "true" else 0
            self.emit(f"LDC  {dest_reg},{int_value}(0)")
        elif node.tipo == "int":
            int_val = int(value)
            self.emit(f"LDC  {dest_reg},{int_val}(0)")
        elif node.tipo == "float":
            # Para flotantes, necesitamos una representación especial
            # Convertir a entero manteniendo la precisión
            try:
                float_val = float(value)
                # Multiplicar por 100 para mantener 2 decimales
                scaled_val = int(float_val * 100)
                self.emit(f"LDC  {dest_reg},{scaled_val}(0)")
            except ValueError:
                self.emit(f"LDC  {dest_reg},0(0)")
        else:
            self.emit(f"LDC  {dest_reg},0(0)")
    
    def visit_variable(self, node, dest_reg):
        """Carga una variable en el registro dest_reg"""
        var_name = node.value
        addr = self.get_var_address(var_name)
        
        if addr is not None:
            self.emit(f"LD  {dest_reg},{addr}(5)")
        else:
            self.emit(f"LDC  {dest_reg},0(0)")
    
    def visit_binary_expression(self, node, dest_reg):
        """Evalúa una expresión binaria"""
        if len(node.children) < 2:
            self.emit(f"LDC  {dest_reg},0(0)")
            return
        
        left_node = node.children[0]
        right_node = node.children[1]
        op = node.value
        
        # Evaluar lado izquierdo en registro 1
        self.visit_expression(left_node, 1)
        
        # Guardar resultado izquierdo en temporal
        temp_addr = self.new_temp_addr()
        self.emit(f"ST  1,{temp_addr}(5)")
        
        # Evaluar lado derecho en registro 0
        self.visit_expression(right_node, 0)
        
        # Cargar izquierdo en registro 1
        self.emit(f"LD  1,{temp_addr}(5)")
        
        # Aplicar operación
        if op == '+':
            self.emit(f"ADD  {dest_reg},1,0")
        elif op == '-':
            self.emit(f"SUB  {dest_reg},1,0")
        elif op == '*':
            self.emit(f"MUL  {dest_reg},1,0")
        elif op == '/':
            self.emit(f"DIV  {dest_reg},1,0")
        elif op == '%':
            self.emit(f"MOD  {dest_reg},1,0")
        elif op == '^':
            # Para exponente, usar multiplicación (simplificado para ^2)
            self.emit(f"MUL  {dest_reg},1,1")
        elif op in ['<', '<=', '>', '>=', '==', '!=']:
            # Operaciones relacionales
            # left is in R1, right is in R0
            if op in ['>', '>=']:
                self.emit(f"SUB  0,0,1") # R0 = R0 - R1 = B - A
            else:
                self.emit(f"SUB  0,1,0") # R0 = R1 - R0 = A - B

            jump_instr = ""
            if op == '<': jump_instr = "JLT"
            elif op == '<=': jump_instr = "JLE"
            elif op == '>': jump_instr = "JLT" # because of B-A
            elif op == '>=': jump_instr = "JLE" # because of B-A
            elif op == '==': jump_instr = "JEQ"
            elif op == '!=': jump_instr = "JNE"
            
            self.emit(f"{jump_instr}  0,4(7)") # if true, jump 4 instructions to the true case

            # False case
            self.emit(f"LDC  {dest_reg},0(0)")
            # Unconditional jump to end (2 instructions)
            self.emit("LDC 3,0(0)")
            self.emit("JEQ 3,2(7)") # Jump over the true case
            
            # True case
            self.emit(f"LDC  {dest_reg},1(0)")
    
    def visit_unary_operation(self, node, dest_reg):
        """Evalúa una operación unaria"""
        if len(node.children) < 1:
            self.emit(f"LDC  {dest_reg},0(0)")
            return
        
        operand_node = node.children[0]
        op = node.value
        
        # Evaluar operando
        self.visit_expression(operand_node, dest_reg)
        
        if op == '++':
            self.emit(f"LDC  1,1(0)")
            self.emit(f"ADD  {dest_reg},{dest_reg},1")
        elif op == '--':
            self.emit(f"LDC  1,1(0)")
            self.emit(f"SUB  {dest_reg},{dest_reg},1")
        elif op == '!':
            label_true = self.new_label()
            label_end = self.new_label()
            
            self.emit(f"JEQ  {dest_reg},0,2(7)")
            # Si no es 0, es falso
            self.emit(f"LDC  {dest_reg},0(0)")
            self.emit(f"LDA  7,{label_end}(7)")
            
            # Marcar posición para verdadero
            self.patch_label(label_true, len(self.code))
            self.emit(f"LDC  {dest_reg},1(0)")
            
            # Marcar final
            self.patch_label(label_end, len(self.code))
    
    def visit_logical_operation(self, node, dest_reg):
        """Evalúa una operación lógica con cortocircuito"""
        if len(node.children) < 2:
            self.emit(f"LDC  {dest_reg},0(0)")
            return
        
        left_node = node.children[0]
        right_node = node.children[1]
        op = node.value
        
        if op == '&&':
            # Si el izquierdo es falso (0), el resultado es falso y ya está en dest_reg.
            # Se salta la evaluación del operando derecho.
            self.visit_expression(left_node, dest_reg)
            jump_to_end = len(self.code)
            self.emit(f"JEQ  {dest_reg}, <placeholder_end>") # Jump if false
            
            # Si el izquierdo es verdadero, el resultado es el valor del derecho.
            self.visit_expression(right_node, dest_reg)

            # Se parchea el salto.
            end_addr = len(self.code)
            offset = end_addr - jump_to_end
            self.code[jump_to_end] = f"    JEQ  {dest_reg},{offset}(7)"

        elif op == '||':
            # Si el izquierdo es verdadero (no 0), el resultado es verdadero y ya está en dest_reg.
            # Se salta la evaluación del operando derecho.
            self.visit_expression(left_node, dest_reg)
            jump_to_end = len(self.code)
            self.emit(f"JNE  {dest_reg}, <placeholder_end>") # Jump if true
            
            # Si el izquierdo es falso, el resultado es el valor del derecho.
            self.visit_expression(right_node, dest_reg)

            # Se parchea el salto.
            end_addr = len(self.code)
            offset = end_addr - jump_to_end
            self.code[jump_to_end] = f"    JNE  {dest_reg},{offset}(7)"
    
    def visit_if(self, node):
        """Visita una estructura if-then-else"""
        if len(node.children) < 2:
            return
        
        # Evaluar condición
        cond_node = node.children[0]
        self.visit_expression(cond_node, 0)
        
        # Salto condicional a la sección else
        jump_to_else_index = len(self.code)
        self.emit("JEQ  0, <placeholder_else>")
        
        # Bloque then
        then_block = node.children[1]
        self.visit_statement(then_block)
        
        jump_to_end_index = -1
        if len(node.children) > 2: # has else
            # Salto incondicional al final
            jump_to_end_index = len(self.code)
            self.emit("LDC  1,0(0)")
            self.emit("JEQ  1, <placeholder_end>")
            
        # Dirección del else
        else_addr = len(self.code)
        offset = else_addr - jump_to_else_index
        self.code[jump_to_else_index] = f"    JEQ  0,{offset}(7)"
        
        # Bloque else
        if len(node.children) > 2:
            else_block = node.children[2]
            if else_block.node_type == "Else" and else_block.children:
                self.visit_statement(else_block.children[0])
            else:
                self.visit_statement(else_block)

        # Dirección final
        end_addr = len(self.code)
        if jump_to_end_index != -1:
            offset_end = end_addr - (jump_to_end_index + 1)
            self.code[jump_to_end_index+1] = f"    JEQ  1,{offset_end}(7)"
    
    def visit_while(self, node):
        """Visita una estructura while"""
        if len(node.children) < 2:
            return
        
        label_start = len(self.code)
        
        # Evaluar condición
        cond_node = node.children[0]
        self.visit_expression(cond_node, 0)
        
        # Saltar al final si es falso
        jump_to_end_index = len(self.code)
        self.emit("JEQ 0, <placeholder_end>")

        # Cuerpo del while
        body_block = node.children[1]
        self.visit_statement(body_block)
        
        # Salto incondicional al inicio
        self.emit("LDC 1,0(0)")
        jump_to_start_offset = label_start - (len(self.code))
        self.emit(f"JEQ 1,{jump_to_start_offset}(7)")

        # Final
        end_addr = len(self.code)
        offset = end_addr - jump_to_end_index
        self.code[jump_to_end_index] = f"    JEQ  0,{offset}(7)"
    
    def visit_dowhile(self, node):
        """Visita una estructura do-while"""
        if len(node.children) < 2:
            return
        
        label_start = len(self.code)
        
        # Cuerpo del do-while
        body_block = node.children[0]
        self.visit_statement(body_block)
        
        # Evaluar condición
        cond_node = node.children[1]
        self.visit_expression(cond_node, 0)
        
        # Repetir si condición es verdadera
        jump_to_start_offset = label_start - len(self.code)
        self.emit(f"JNE  0,{jump_to_start_offset}(7)")
    
    def visit_input(self, node):
        """Visita una sentencia de entrada (cin >> variable)"""
        if len(node.children) >= 3:
            id_node = node.children[2]
            if id_node and id_node.node_type == "ID":
                var_name = id_node.value
                addr = self.get_var_address(var_name)
                
                if addr is not None:
                    self.emit(f"IN  0,0,0")
                    self.emit(f"ST  0,{addr}(5)")
    
    def visit_output(self, node):
        """Visita una sentencia de salida (cout << expresión)"""
        if len(node.children) >= 3:
            output_expr_node = node.children[2]  # Este es el nodo de salida (SALIDA_EXPR)
            
            # Si es un nodo SALIDA_EXPR, extraer la expresión real
            if output_expr_node.node_type == "SALIDA_EXPR":
                if output_expr_node.children:
                    # El primer hijo es la expresión real
                    real_expr = output_expr_node.children[0]
                    self.visit_output_expression(real_expr)
                else:
                    self.emit(f"LDC  0,0(0)")
                    self.emit(f"LDC  1,1(0)")  # Código para número
                    self.emit(f"OUT  1,0,0")
                    self.emit(f"OUT  0,0,0")
            else:
                # Si no es SALIDA_EXPR, asumir que es la expresión directamente
                self.visit_output_expression(output_expr_node)
    
    def visit_output_expression(self, node):
        if node.node_type == "CADENA":
            # Para cadenas, usar código especial 0 antes de cada carácter
            self.visit_string_output_with_special_code(node)
        else:
            # Para expresiones numéricas, usar código especial 1
            self.visit_numeric_output_with_special_code(node)
            
    def visit_string_output_with_special_code(self, node):
        """Genera código para imprimir una cadena con código especial 0"""
        if node.value.startswith('"') and node.value.endswith('"'):
            string_val = node.value[1:-1]
        elif node.value.startswith("'") and node.value.endswith("'"):
            string_val = node.value[1:-1]
        else:
            string_val = node.value
        
        # Manejar caracteres de escape
        string_val = string_val.replace('\\n', '\n').replace('\\t', '\t').replace('\\\\', '\\')
        
        # Para cada carácter, enviar código 0 seguido del valor ASCII
        for char in string_val:
            ascii_val = ord(char)
            self.emit(f"LDC  0,0(0)")    # Código 0 = cadena
            self.emit(f"OUT  0,0,0")
            self.emit(f"LDC  0,{ascii_val}(0)")
            self.emit(f"OUT  0,0,0")
    
    def visit_numeric_output_with_special_code(self, node):
        """Genera código para imprimir un valor numérico con código especial 1"""
        self.visit_expression(node, 0)
        self.emit(f"LDC  1,1(0)")    # Código 1 = número
        self.emit(f"OUT  1,0,0")
        self.emit(f"OUT  0,0,0")     # Valor numérico
    
    def visit_string_output(self, node):
        """Ahora usa la nueva función para cadenas"""
        self.visit_string_output_with_special_code(node)
    
class ASTNode:
    def __init__(self, node_type, value=None, line=None, col=None, error=False):
        self.node_type = node_type
        self.value = value
        self.line = line
        self.col = col
        self.children = []
        self.is_error = error
        self.tipo = None
        self.calculated_value = None
    
    def add_child(self, child_node):
        self.children.append(child_node)
    
    def __repr__(self, level=0):
        prefix = "🚫 " if self.is_error else ""
        ret = "  " * level + f"{self.node_type} (id: {id(self)})"
        if self.value:
            ret += f": {self.value}"
        if self.line and self.col:
            ret += f" [Línea: {self.line}, Col: {self.col}]"
        if hasattr(self, 'tipo') and self.tipo:
            ret += f" [tipo: {self.tipo}]"
        if hasattr(self, 'calculated_value') and self.calculated_value is not None:
            ret += f" [Valor Calculado: {self.calculated_value}]"
        ret += "\n"
        
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret
    
class ASTViewer(QTreeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabel("Árbol Sintáctico Abstracto")
        self.setColumnCount(6)
        self.setHeaderLabels(["Nodo", "Valor", "Tipo", "Valor Calculado", "Línea", "Columna"])
        
        # Ajustar el ancho de las columnas
        self.setColumnWidth(0, 250)  # Nodo
        self.setColumnWidth(1, 100)  # Valor
        self.setColumnWidth(2, 80)   # Tipo
        self.setColumnWidth(3, 100)  # Valor Calculado
        self.setColumnWidth(4, 60)   # Línea
        self.setColumnWidth(5, 60)   # Columna
    
    def display_ast(self, ast_root):
        self.clear()
        if ast_root:
            self._add_tree_item(None, ast_root)
            self.expandAll()
    
    def _add_tree_item(self, parent_item, ast_node):
        # Crear texto para cada columna
        node_text = ast_node.node_type
        value_text = ast_node.value if ast_node.value else ""
        tipo_text = ast_node.tipo if hasattr(ast_node, 'tipo') and ast_node.tipo else ""
        calculated_value_text = str(ast_node.calculated_value) if hasattr(ast_node, 'calculated_value') and ast_node.calculated_value is not None else ""
        line_text = str(ast_node.line) if ast_node.line else ""
        col_text = str(ast_node.col) if ast_node.col else ""
        
        # Crear el ítem del árbol
        item = QTreeWidgetItem([node_text, value_text, tipo_text, calculated_value_text, line_text, col_text])
        
        if ast_node.is_error:
            # Fondo rojo para nodos con errores
            for i in range(6):  # Aplicar a todas las columnas
                item.setBackground(i, QColor(255, 200, 200))  # Rojo claro
                item.setForeground(i, QColor(255, 0, 0))       # Texto rojo
            item.setToolTip(0, "Este nodo contiene un error sintáctico")
            
        # Configurar estilo para nodos especiales
        if ast_node.node_type == "PROGRAMA":
            item.setBackground(0, QColor(30, 144, 255))  # Azul para el nodo raíz
            item.setForeground(0, QColor(255, 255, 255))
        elif ast_node.node_type in ["MAIN", "LISTA_DECLARACION"]:
            item.setBackground(0, QColor(70, 130, 180))  # Azul acero para nodos importantes
            item.setForeground(0, QColor(255, 255, 255))
        
        if hasattr(ast_node, 'tipo') and ast_node.tipo:
            item.setBackground(2, QColor(144, 238, 144))  # Verde claro para tipos
            item.setToolTip(2, f"Tipo inferido: {ast_node.tipo}")
        
        if parent_item:
            parent_item.addChild(item)
        else:
            self.addTopLevelItem(item)
        
        for child in ast_node.children:
            self._add_tree_item(item, child)

class CodeEditor(QPlainTextEdit):
    def __init__(self, file_path=None):
        super().__init__()
        self.file_path = file_path
        self.line_number_area = LineNumberArea(self)
        self.blockCountChanged.connect(self.update_line_number_area_width)
        self.updateRequest.connect(self.update_line_number_area)
        self.cursorPositionChanged.connect(self.update_cursor_position)
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setFont(QFont("Consolas", 12))
        self.highlighter = SyntaxHighlighter(self.document())
        self.update_line_number_area_width(0)

    def line_number_area_width(self):
        digits = len(str(self.blockCount()))
        return 30 + self.fontMetrics().width('9') * digits

    def update_line_number_area_width(self, _):
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def update_line_number_area(self, rect, dy):
        if dy:
            self.line_number_area.scroll(0, dy)
        else:
            self.line_number_area.update(0, rect.y(), self.line_number_area.width(), rect.height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.line_number_area.setGeometry(cr.left(), cr.top(), self.line_number_area_width(), cr.height())

    def update_cursor_position(self):
        line = self.textCursor().blockNumber() + 1
        col = self.textCursor().columnNumber() + 1  
        self.window().statusBar().showMessage(f"Línea: {line}, Columna: {col}")

class LineNumberArea(QWidget):
    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(event.rect(), Qt.lightGray)
        block = self.editor.firstVisibleBlock()
        block_number = block.blockNumber()
        top = self.editor.blockBoundingGeometry(block).translated(self.editor.contentOffset()).top()
        bottom = top + self.editor.blockBoundingRect(block).height()

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                painter.setPen(Qt.black)
                painter.drawText(0, int(top), self.width() - 5, self.editor.fontMetrics().height(),
                                 Qt.AlignRight, str(block_number + 1))

            block = block.next()
            top = bottom
            bottom = top + self.editor.blockBoundingRect(block).height()
            block_number += 1

class CompilerIDE(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.compiler_path = "compilador.exe"
        self.interpreter_thread = None
        
    def initUI(self):
        self.setWindowTitle("BNV Code")
        self.setGeometry(100, 100, 1200, 800)

        # Widgets principales
        self.editor = CodeEditor()
        self.setCentralWidget(self.editor)
        
        # Widget principal: pestañas
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self.close_tab)
        self.setCentralWidget(self.tab_widget)
        
        # Paneles de resultados
        self.create_output_panels()
        
        # Barra de estado
        self.statusBar().setStyleSheet("color: white;")
        self.statusBar().showMessage("Listo")

        # Menús
        self.create_menus()
        
        # Barra de herramientas
        self.create_toolbar()
        
        # Aplicar CSS
        self.apply_styles()
        
        # Agregar primera pestaña
        self.new_file()
        
    # Funciones de análisis lexico
    
    # Esta función se encarga de ejecutar el análisis léxico
    # y mostrar los resultados en el panel correspondiente.
    
    # También maneja la creación de archivos de salida para
    # el archivo de tokens y archivo de errores encontrados durante el análisis.
    
    def run_lexical(self):
        editor = self.get_current_editor() # Obtener el editor actual
        if editor: # Si hay un editor abierto
            code = editor.toPlainText() # Obtener el texto del editor
            analyzer = LexicalAnalyzer() # Crear una instancia del analizador léxico
            tokens, errors = analyzer.analyze(code) # Analizar el código
            
            # Mostrar tokens
            self.lexical_output.clear() # Limpiar el panel de salida léxica
            for token in tokens: # Recorrer los tokens generados
                self.lexical_output.append(f"Token: {token[0]} ({token[1]})") # Mostrar el token y su tipo
                
            # Guardar en tokens.txt
            try:
                current_dir = os.getcwd()  # Obtener directorio actual
                tokens_path = os.path.join(current_dir, "tokens.txt") # Crear ruta para el archivo de tokens
                with open(tokens_path, "w", encoding="utf-8") as f: # Abrir o crear el archivo en modo escritura
                    for token in tokens:
                        # Formato: tipo,lexema,línea,columna
                        f.write(f"{token[1]}¬{token[0]}¬{token[2]}¬{token[3]}\n") # Guardar los tokens en el archivo
            except Exception as e: # Manejar errores al guardar
                self.error_list.addItem(f"Error al guardar tokens: {str(e)}") # Mostrar error en el panel de errores
            
            self.error_list.clear() # Limpiar el panel de errores
            error_content = [] # Lista para almacenar errores
            for error in errors: # Recorrer los errores encontrados
                error_msg = f"Error en línea {error['line']}, col {error['col']}: '{error['value']}'" # Plantilla de errores
                self.error_list.addItem(error_msg) # Mostrar error en el panel de errores
                error_content.append(error_msg) # Agregar error a la lista de errores
                
            # Guardar en errors.txt
            try:
                errors_path = os.path.join(current_dir, "errors.txt") # Crear ruta para el archivo de errores
                with open(errors_path, "w", encoding="utf-8") as f: # Abrir o crear el archivo en modo escritura
                    f.write("\n".join(error_content)) # Guardar errores en el archivo
            except Exception as e:
                self.error_list.addItem(f"Error al guardar errores: {str(e)}")
            
            # Resaltar errores en el editor
            editor.highlighter.error_positions = {} # Limpiar posiciones de errores
            for error in errors: # Recorrer errores encontrados
                line = error['line'] # Obtener número de línea
                col_start = error['col'] - 1 # Obtener posición de inicio del error
                length = error['length'] # Obtener longitud del error
                if line not in editor.highlighter.error_positions: # Si no hay errores en la línea 
                    editor.highlighter.error_positions[line] = [] # Crear lista de errores
                editor.highlighter.error_positions[line].append((col_start, length)) # Agregar error a la lista
            editor.highlighter.rehighlight() # Reaplicar resaltado para mostrar errores
    
    # Funciones de manejo de pestañas
    # Esta función se encarga de crear nuevas pestañas en el editor para cada archivo abierto.
    def create_new_tab(self, content="", file_path=None): 
        editor = CodeEditor(file_path) # Crear un nuevo editor de texto
        self.tab_widget.addTab(editor, "Nuevo" if not file_path else file_path.split("/")[-1]) # Agregar pestaña
        self.tab_widget.setCurrentWidget(editor) # Establecer pestaña actual
        editor.setPlainText(content) # Establecer contenido del editor
        return editor # Retornar el editor creado
    
    # Esta función se encarga de cerrar una pestaña en el editor.
    def close_tab(self, index): # Cerrar pestaña
        widget = self.tab_widget.widget(index) # Obtener el widget de la pestaña
        if widget.document().isModified(): # Si el documento ha sido modificado
            reply = QMessageBox.question(self, 'Guardar cambios', 
                "¿Deseas guardar los cambios?", QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel) # Mostrar mensaje de guardar cambios
            
            if reply == QMessageBox.Yes: # Si se desea guardar
                self.save_file() # Guardar archivo
            elif reply == QMessageBox.Cancel: # Si se cancela
                return # Cancelar cierre de pestaña
        
        self.tab_widget.removeTab(index) # Cerrar pestaña
    
    # Función para obtener el editor actual
    # Esta función se encarga de obtener el editor actual en la pestaña activa.
    # Se utiliza para realizar operaciones como guardar o compilar el archivo abierto.
    def get_current_editor(self):
        return self.tab_widget.currentWidget()
    
    # Función para crear los paneles de resultados
    def create_output_panels(self):
        # Panel de pestañas para resultados
        self.tabs = QTabWidget()
        self.output = QTextEdit()
        self.lexical_output = QTextEdit()
        self.lexical_output.setReadOnly(True)
        self.syntax_output = QTextEdit()
        self.semantic_output = QTextEdit()
        self.intermediate_code = QTextEdit()
        self.symbol_table = QTableWidget()
        self.error_list = QListWidget()
        self.execution_output = QTextEdit()
        self.execution_output.setReadOnly(True)
        self.execution_output.setFont(QFont("Consolas", 10))
        
        # Crear visores para AST normal y AST anotado
        self.ast_viewer = ASTViewer()
        self.ast_anotado_viewer = ASTViewer() 
        
        # Configuración de los paneles y sus nombres
        self.tabs.addTab(self.output, "Salida")
        self.tabs.addTab(self.lexical_output, "Léxico")
        self.tabs.addTab(self.syntax_output, "Sintáctico")
        self.tabs.addTab(self.ast_viewer, "Árbol AST")
        self.tabs.addTab(self.semantic_output, "Semántico")
        self.tabs.addTab(self.ast_anotado_viewer, "Árbol AST Anotado")
        self.tabs.addTab(self.intermediate_code, "Código Intermedio")
        self.tabs.addTab(self.symbol_table, "Tabla de Símbolos")
        self.tabs.addTab(self.execution_output, "Ejecución")
        self.tabs.addTab(self.error_list, "Errores")
        
        # Configuración del panel de pestañas
        self.results_dock = QDockWidget("Resultados", self)
        self.results_dock.setWidget(self.tabs) # Establecer el panel de pestañas como widget del dock
        self.results_dock.setAllowedAreas(Qt.AllDockWidgetAreas) # Permitir que el dock se mueva
        self.results_dock.setFeatures(QDockWidget.DockWidgetClosable | 
                                    QDockWidget.DockWidgetMovable | 
                                    QDockWidget.DockWidgetFloatable) # Permitir que el dock sea flotante
        self.addDockWidget(Qt.BottomDockWidgetArea, self.results_dock) # Agregar el dock a la parte inferior de la ventana
        
    # Función para crear los menús
    # Esta función se encarga de crear los menús de la aplicación.
    # Se crean los menús de Archivo, Vista y Compilar.
    def create_menus(self):
        menubar = self.menuBar() # Crear la barra de menú
        
        # Menú Archivo
        file_menu = menubar.addMenu("Archivo")
        # Acciones del menú de archivo
        file_menu.addAction("Nuevo", self.new_file, "Ctrl+N")
        file_menu.addAction("Abrir", self.open_file, "Ctrl+O")
        file_menu.addAction("Guardar", self.save_file, "Ctrl+S")
        file_menu.addAction("Guardar como", self.save_file_as)
        
        # Menu Vista
        view_menu = menubar.addMenu("Vista")
        # Acciones del menú de vista
        view_menu.addAction(self.results_dock.toggleViewAction())
        self.results_dock.toggleViewAction().setText("Mostrar/Ocultar Resultados")
        
        # Menú Compilar
        compile_menu = menubar.addMenu("Compilar")
        # Acciones del menú de compilar (cada uno de los análisis)
        compile_menu.addAction("Análisis Léxico", self.run_lexical)
        compile_menu.addAction("Análisis Sintáctico", self.run_syntax)
        compile_menu.addAction("Análisis Semántico", self.run_semantic)
        compile_menu.addAction("Generar Código Intermedio", self.run_intermediate)
        compile_menu.addAction("Ejecutar", self.run_execution)
        execute_menu = menubar.addMenu("Ejecutar")
        execute_menu.addAction("Ejecutar Código", self.execute_p_code)
        execute_menu.addAction("Detener Ejecución", self.stop_execution)
    
    # Función para crear la barra de herramientas
    def create_toolbar(self):
        # Barra de herramientas
        toolbar = self.addToolBar("Herramientas")
        
        # Icono de Abrir
        abrir_icon = self.style().standardIcon(QStyle.SP_DialogOpenButton)
        toolbar.addAction(abrir_icon, "Abrir", self.open_file)
        
        # Icono de Guardar
        guardar_icon = self.style().standardIcon(QStyle.SP_DialogSaveButton)
        toolbar.addAction(guardar_icon, "Guardar", self.save_file)
        
        # Separador
        toolbar.addSeparator()
        
        # Acciones de análisis (texto)
        toolbar.addAction("Léxico", self.run_lexical)
        toolbar.addAction("Sintáctico", self.run_syntax)
        toolbar.addAction("Semántico", self.run_semantic)
        toolbar.addAction("Codigo intermedio", self.run_intermediate)
        
        # Icono de Ejecutar
        execute_icon = self.style().standardIcon(QStyle.SP_MediaPlay)
        stop_icon = self.style().standardIcon(QStyle.SP_MediaStop)
        
        toolbar.addAction(execute_icon, "Ejecutar", self.execute_p_code)
        toolbar.addAction(stop_icon, "Detener", self.stop_execution)
        
    def execute_p_code(self):
        """Ejecuta el código P generado"""
        # Obtener código P del panel de código intermedio o del archivo
        codigo_p = self.intermediate_code.toPlainText()
        
        if not codigo_p.strip():
            # Intentar cargar del archivo
            try:
                with open("programa.P", "r", encoding="utf-8") as f:
                    codigo_p = f.read()
            except FileNotFoundError:
                QMessageBox.warning(self, "Error", 
                    "No hay código P para ejecutar. Genere código intermedio primero.")
                return
        
        # Limpiar panel de ejecución
        self.execution_output.clear()
        self.execution_output.append("=== INICIANDO EJECUCIÓN ===\n")
        
        # Detener ejecución anterior si existe
        if self.interpreter_thread and self.interpreter_thread.isRunning():
            self.stop_execution()
        
        # Crear y configurar hilo del intérprete
        self.interpreter_thread = PInterpreterThread(codigo_p)
        self.interpreter_thread.output_signal.connect(self.handle_interpreter_output)
        self.interpreter_thread.input_request_signal.connect(self.handle_input_request)
        self.interpreter_thread.execution_finished.connect(self.handle_execution_finished)
        self.interpreter_thread.error_signal.connect(self.handle_interpreter_error)
        
        # Configurar botón de detener
        self.statusBar().showMessage("Ejecutando código...")
        
        # Iniciar ejecución
        self.interpreter_thread.start()
        
        # Cambiar a pestaña de ejecución
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Ejecución":
                self.tabs.setCurrentIndex(i)
                break
    
    def stop_execution(self):
        """Detiene la ejecución del código"""
        if self.interpreter_thread and self.interpreter_thread.isRunning():
            self.interpreter_thread.should_stop = True
            self.interpreter_thread.wait()
            self.execution_output.append("\n--- Ejecución detenida por el usuario ---")
            self.statusBar().showMessage("Ejecución detenida")
            
    def handle_interpreter_output(self, output):
        """Maneja la salida del intérprete"""
        self.execution_output.insertPlainText(output)
        # Auto-desplazar al final
        cursor = self.execution_output.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.execution_output.setTextCursor(cursor)
        self.execution_output.ensureCursorVisible()
    
    def handle_input_request(self, message):
        """Maneja solicitudes de entrada del intérprete"""
        text, ok = QInputDialog.getText(self, "Entrada de Ejecución", message)
        if ok and text is not None:
            self.interpreter_thread.provide_input(text)
        else:
            self.interpreter_thread.provide_input("0")
    
    def handle_execution_finished(self):
        """Maneja la finalización de la ejecución"""
        self.execution_output.append("\n=== EJECUCIÓN FINALIZADA ===")
        self.statusBar().showMessage("Ejecución completada")
        
        # Mostrar estado final
        self.execution_output.append("\nEstado final de registros:")
        for i in range(8):
            self.execution_output.append(f"  R{i}: {self.interpreter_thread.interpreter.registers[i] if hasattr(self.interpreter_thread, 'interpreter') else 'N/A'}")
    
    def handle_interpreter_error(self, error_message):
        """Maneja errores del intérprete"""
        self.execution_output.append(f"\n⚠️ ERROR: {error_message}")
        self.statusBar().showMessage(f"Error: {error_message}")
    
    # Función para aplicar estilos CSS
    # Esta función se encarga de aplicar estilos CSS a la interfaz gráfica.
    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #ffffff;
            }
            QPlainTextEdit, QTextEdit {
                background-color: #1e1e1e;
                color: #dcdcdc;
                border: 1px solid #3c3c3c;
            }
            QTabWidget::pane {
                border: 1px solid #3c3c3c;
            }
            QDockWidget {
                background: #2b2b2b;
                color: #ffffff;
            }
            QMenuBar {
                background-color: #3c3c3c;
                color: #ffffff;
            }
            QToolBar {
                background:rgb(235, 235, 235);
                color: #ffffff;
                border: none;
            }
        """)
        
    # Funciones de manejo de archivos
    
    # Funcion de nuevo archivo
    # Esta función se encarga de crear un nuevo archivo en el editor.
    def new_file(self):
        self.create_new_tab() # Crear nueva pestaña
    
    # Función de abrir archivo
    # Esta función se encarga de abrir un archivo existente en el editor.
    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Abrir archivo", "", "Archivos de texto (*.txt)") # Abrir diálogo de selección de archivo
        if filename: # Si se selecciona un archivo
            with open(filename, 'r') as f: # Abrir el archivo en modo lectura
                content = f.read() # Leer el contenido del archivo
            editor = self.create_new_tab(content, filename) # Crear nueva pestaña con el contenido del archivo
            editor.file_path = filename # Establecer la ruta del archivo
    
    # Función de guardar archivo
    def save_file(self): 
        editor = self.get_current_editor() # Obtener el editor actual
        if editor: # Si hay un editor abierto
            if editor.file_path: # Si el archivo tiene una ruta
                with open(editor.file_path, 'w') as f: # Abrir el archivo en modo escritura
                    f.write(editor.toPlainText()) # Guardar el contenido del editor en el archivo
                editor.document().setModified(False) # Marcar el documento como no modificado
            else: # Si no hay ruta de archivo
                self.save_file_as() # Llamar a la función de guardar como
    
    # Función de guardar como
    def save_file_as(self):
        editor = self.get_current_editor() # Obtener el editor actual
        if editor: # Si hay un editor abierto
            filename, _ = QFileDialog.getSaveFileName(self, "Guardar como", "", "Archivos de texto (*.txt)") # Abrir diálogo de guardar como
            if filename: # Si se selecciona un archivo
                with open(filename, 'w') as f: # Abrir el archivo en modo escritura
                    f.write(editor.toPlainText()) # Guardar el contenido del editor en el archivo
                editor.file_path = filename # Establecer la ruta del archivo
                self.tab_widget.setTabText(self.tab_widget.indexOf(editor), filename.split("/")[-1]) # Cambiar el nombre de la pestaña
                editor.document().setModified(False) # Marcar el documento como no modificado
    
    # Funciones de compilación (No usadas de forma activa actualmente)
    def run_compiler(self, phase):
        editor = self.get_current_editor() 
        if editor:
            if not editor.file_path:
                self.save_file()
            if editor.file_path:
                if not self.current_file:
                    self.save_file()
                if self.current_file:
                    process = QProcess(self)
                    command = f'"{self.compiler_path}" --{phase} "{self.current_file}"'
            
            def handle_output():
                output = process.readAllStandardOutput().data().decode()
                error = process.readAllStandardError().data().decode()
                
                if phase == "lex":
                    self.lexical_output.setPlainText(output)
                elif phase == "syntax":
                    self.syntax_output.setPlainText(output)
                
                if error:
                    self.error_list.addItem(f"Error en {phase}: {error}")
            
            process.readyReadStandardOutput.connect(handle_output)
            process.readyReadStandardError.connect(handle_output)
            process.start(command)
        
    def run_syntax(self):
        # Obtener tokens del archivo
        tokens = []
        try:
            with open("tokens.txt", "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split('¬')
                    if len(parts) == 4:
                        tokens.append((parts[0], parts[1], int(parts[2]), int(parts[3])))
        except Exception as e:
            self.error_list.addItem(f"Error al leer tokens: {str(e)}")
            return
    
        # Ejecutar parser
        parser = Parser(tokens)
        ast, errors = parser.parse()
        self.ast = ast  # Guardar AST para análisis semántico
    
        # Mostrar errores
        self.error_list.clear()
        for error in errors:
            # Verificar si el error tiene información de posición
            if 'line' in error and 'col' in error:
                self.error_list.addItem(
                    f"Error sintáctico en línea {error['line']}, col {error['col']}: {error['message']}"
                )
            else:
                self.error_list.addItem(f"Error sintáctico: {error['message']}")
        
        # Mostrar el AST normal
        self.ast_viewer.display_ast(ast)
    
        # Resaltar nodos con errores
        self.highlight_error_nodes(ast)
        
        ast_tab_index = -1
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Árbol AST":
                ast_tab_index = i
                break
        if ast_tab_index != -1:
            self.tabs.setCurrentIndex(ast_tab_index)
        
        # Guardar AST en archivo
        if ast:
            with open("ast.txt", "w", encoding="utf-8") as f:
                f.write(ast.__repr__())
                
        self.error_list.clear()
        error_content = []
        
        try:
            with open("errors.txt", "r", encoding="utf-8") as f:
                error_content = f.read().splitlines()
        except:
            pass
        
        for error in errors:
            if 'line' in error and 'col' in error:
                error_msg = f"Error sintáctico en línea {error['line']}, col {error['col']}: {error['message']}"
            else:
                error_msg = f"Error sintáctico: {error['message']}"
        
            self.error_list.addItem(error_msg)
            error_content.append(error_msg)  # Agregar a la lista de errores
    
        # Guardar TODOS los errores en errors.txt
        try:
            current_dir = os.getcwd()
            errors_path = os.path.join(current_dir, "errors.txt")
            with open(errors_path, "w", encoding="utf-8") as f:
                f.write("\n".join(error_content))
        except Exception as e:
            self.error_list.addItem(f"Error al guardar errores: {str(e)}")
    
    def highlight_error_nodes(self, node):
        """Recorrer AST y marcar nodos con errores"""
        if node.is_error:
            pass
        for child in node.children:
            self.highlight_error_nodes(child)
        
    def run_semantic(self):
        """Ejecutar análisis semántico"""
        if not hasattr(self, 'ast') or self.ast is None:
            self.error_list.addItem("Error: debe ejecutar análisis sintáctico primero")
            return
    
        # Ejecutar analizador semántico
        analyzer = SemanticAnalyzer()
        errores_semanticos = analyzer.analizar(self.ast)

        # Mostrar tabla de símbolos
        self.mostrar_tabla_simbolos(analyzer.tabla_simbolos)

        # Mostrar errores semánticos
        self.error_list.clear()
        for error in errores_semanticos:
            error_msg = f"Error semántico en línea {error['linea']}, col {error['columna']}: {error['mensaje']}"
            self.error_list.addItem(error_msg)
    
        # Guardar errores en archivo
        self.guardar_errores_semanticos(errores_semanticos)

        # Guardar tabla de símbolos
        self.guardar_tabla_simbolos(analyzer.tabla_simbolos)

        # Mostrar AST anotado
        self.mostrar_ast_anotado(analyzer.ast_anotado)

        # Mostrar resumen
        if errores_semanticos:
            self.semantic_output.setPlainText(f"Se encontraron {len(errores_semanticos)} errores semánticos")
        else:
            self.semantic_output.setPlainText("Análisis semántico exitoso. No se encontraron errores.")

        # Cambiar a pestaña de AST anotado
        ast_anotado_tab_index = -1
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == "Árbol AST Anotado":
                ast_anotado_tab_index = i
                break
        if ast_anotado_tab_index != -1:
            self.tabs.setCurrentIndex(ast_anotado_tab_index)
        # Guardar el AST anotado para uso posterior
        self.ast_anotado = analyzer.ast_anotado

    def mostrar_tabla_simbolos(self, tabla_simbolos):
        """Mostrar tabla de símbolos en el panel correspondiente"""
        self.symbol_table.clear()
        # MODIFICADO: Reducir a 4 columnas (eliminar Ámbito, Apariciones, Desplazamiento)
        self.symbol_table.setColumnCount(4)
        # MODIFICADO: Eliminar "Ámbito", "Apariciones" y "Desplazamiento" de los encabezados
        self.symbol_table.setHorizontalHeaderLabels(["Nombre", "Tipo", "Líneas", "Dirección"])

        fila = 0
        for clave, info in tabla_simbolos.symbols.items():
            nombre = clave.split('::')[1]
            self.symbol_table.insertRow(fila)
            self.symbol_table.setItem(fila, 0, QTableWidgetItem(nombre))
            self.symbol_table.setItem(fila, 1, QTableWidgetItem(info['tipo']))
            
            # MODIFICADO: Eliminar la columna de Ámbito (índice 2)
            # Usar el nuevo método para mostrar líneas con conteo
            lineas_str = tabla_simbolos.obtener_lineas_con_apariciones(nombre, info['ambito'])
            self.symbol_table.setItem(fila, 2, QTableWidgetItem(lineas_str))
            
            # MODIFICADO: Eliminar la columna de Apariciones (índice 4)
            # Mostrar solo dirección (índice 3 ahora)
            self.symbol_table.setItem(fila, 3, QTableWidgetItem(str(info['direccion'])))
            
            fila += 1

        self.symbol_table.resizeColumnsToContents()

    def guardar_errores_semanticos(self, errores):
        """Guardar errores semánticos en archivo"""
        try:
            with open("errores_semanticos.txt", "w", encoding="utf-8") as f:
                for error in errores:
                    f.write(f"Línea {error['linea']}, Col {error['columna']}: {error['mensaje']}\n")
        except Exception as e:
            self.error_list.addItem(f"Error al guardar errores semánticos: {str(e)}")

    def guardar_tabla_simbolos(self, tabla_simbolos):
        """Guardar tabla de símbolos en archivo"""
        try:
            with open("tabla_simbolos.txt", "w", encoding="utf-8") as f:
                f.write(str(tabla_simbolos))
        except Exception as e:
            self.error_list.addItem(f"Error al guardar tabla de símbolos: {str(e)}")

    def mostrar_ast_anotado(self, ast_anotado):
        """Mostrar AST con anotaciones semánticas en el visor específico"""
        if ast_anotado:
            self.ast_anotado_viewer.display_ast(ast_anotado)
    
        # Guardar AST anotado en archivo
        if ast_anotado:
            with open("ast_anotado.txt", "w", encoding="utf-8") as f:
                f.write(self.ast_a_texto_con_tipos(ast_anotado))

    def ast_a_texto_con_tipos(self, nodo, nivel=0):
        """Convertir AST anotado a texto mostrando tipos"""
        texto = "  " * nivel + f"{nodo.node_type}"
        if nodo.value:
            texto += f": {nodo.value}"
        if hasattr(nodo, 'tipo') and nodo.tipo:
            texto += f" [tipo: {nodo.tipo}]"
        if hasattr(nodo, 'calculated_value') and nodo.calculated_value is not None:
            texto += f" [Valor Calculado: {nodo.calculated_value}]"
        if nodo.line and nodo.col:
            texto += f" [Línea: {nodo.line}, Col: {nodo.col}]"
        texto += "\n"
    
        for hijo in nodo.children:
            texto += self.ast_a_texto_con_tipos(hijo, nivel + 1)
        return texto
        
    def run_intermediate(self):
        """Generar código intermedio P ejecutable a partir del AST anotado"""
        if not hasattr(self, 'ast_anotado') or self.ast_anotado is None:
            self.error_list.addItem("Error: debe ejecutar análisis semántico primero")
            return

        try:
            # Crear generador de código
            generator = CodeGenerator()

            # Generar código ejecutable
            codigo_p = generator.generate(self.ast_anotado)

            # Mostrar en el panel de código intermedio
            self.intermediate_code.clear()
            self.intermediate_code.setPlainText(codigo_p)

            # Guardar en archivo ejecutable programa.P (sin comentarios)
            with open("programa.P", "w", encoding="utf-8") as f:
                f.write(codigo_p)

            # Cambiar a la pestaña de código intermedio
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i) == "Código Intermedio":
                    self.tabs.setCurrentIndex(i)
                    break
        
            self.statusBar().showMessage("Código intermedio ejecutable generado exitosamente en programa.P")

        except Exception as e:
            self.error_list.addItem(f"Error al generar código intermedio: {str(e)}")
            import traceback
            self.intermediate_code.setPlainText(f"Error: {str(e)}\n\n{traceback.format_exc()}")
        
    def run_execution(self):
        """Ejecutar análisis léxico, sintáctico y semántico en orden"""
        # Limpiar resultados anteriores
        self.lexical_output.clear()
        self.syntax_output.clear()
        self.semantic_output.clear()
        self.error_list.clear()
        self.ast_viewer.clear()
        self.ast_anotado_viewer.clear()
        self.symbol_table.clear()
        
        # Ejecutar análisis léxico
        self.run_lexical()
        
        # Verificar si hay errores léxicos
        if self.error_list.count() > 0:
            self.error_list.addItem("No se puede continuar con análisis sintáctico debido a errores léxicos")
            return
            
        # Ejecutar análisis sintáctico
        self.run_syntax()
        
        # Verificar si hay errores sintácticos
        if self.error_list.count() > 0:
            self.error_list.addItem("No se puede continuar con análisis semántico debido a errores sintácticos")
            return
            
        # Ejecutar análisis semántico
        self.run_semantic()
        
        # Mostrar mensaje de finalización
        self.statusBar().showMessage("Ejecución completada")
        

# Función principal
# Esta función se encarga de iniciar la aplicación y mostrar la ventana principal.
if __name__ == '__main__':
    app = QApplication(sys.argv) # Crear la aplicación
    app.setStyle("Fusion") # Establecer el estilo de la aplicación
    ide = CompilerIDE() # Crear la ventana principal
    ide.show() # Mostrar la ventana
    sys.exit(app.exec_()) # Ejecutar la aplicación