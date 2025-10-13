# Compilador BNVCode
# Integrantes:
# - Bryan Misael Morales Martin
# - Naomi Hernandez Romo
# - Ricardo Andres Veloz Hernandez
# 8A TM 

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
        ('PALABRA_RESERVADA', r'\b(if|then|else|end|do|until|while|switch|case|int|float|main|cin|cout)\b', QColor('#569CD6')),
        # Números
        ('NUM_REAL_INCOMPLETO', r'\d+\.(?!\d)', QColor('#FF0000')), # Errores en números reales
        ('NUM_REAL', r'\d+\.\d+', QColor('#1788ff')),
        ('NUM_ENTERO', r'\d+', QColor('#1788ff')),
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
        # Identificadores
        ('IDENTIFICADOR', r'[a-zA-Z_][a-zA-Z0-9_]*', QColor('#ff56e3')),
        # Palabras reservadas
        ('PALABRA_RESERVADA', r'\b(if|then|else|end|do|until|while|switch|case|int|float|main|cin|cout)\b', QColor('#569CD6')),
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
        # Crear nodo de asignación
        assign_node = ASTNode("ASIGNACION")
    
        # Obtener identificador
        token = self.current_token()
        if token and token[0] == 'IDENTIFICADOR':
            id_node = ASTNode("ID", token[1], token[2], token[3])
            assign_node.add_child(id_node)
            self.advance()
        else:
            self.error("Se esperaba un identificador", token)
            return None

        # Agregar operador de asignación
        assign_node.add_child(ASTNode("ASIGN", "=", token[2], token[3]))
    
        # Obtener operador de incremento (++ o --)
        op_token = self.current_token()
        if op_token and op_token[0] == 'OPERADOR_ARIT' and op_token[1] in ['++', '--']:
            op = op_token[1]
            self.advance()
        else:
            self.error("Se esperaba '++' o '--'", op_token)
            return None

        # Crear expresión equivalente: id = id ± 1
        expr_node = ASTNode(op)
    
        # Crear nodo operador con sus operandos como hijos
        op_node = ASTNode("OPERADOR_ARIT", "+" if op == "++" else "-", op_token[2], op_token[3])
        op_node.add_child(ASTNode("ID", token[1], token[2], token[3]))  # Operando izquierdo
        op_node.add_child(ASTNode("NUM_ENTERO", "1", op_token[2], op_token[3]))  # Operando derecho

        expr_node.add_child(op_node)
        assign_node.add_child(expr_node)
        
        # Verificar punto y coma final
        if self.match('SIMBOLOS', ';'):
            self.advance()
        else:
            self.error("Se esperaba ';'", self.current_token())

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
        node = ASTNode("=")

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
        elif self.match('PALABRA_RESERVADA', 'true') or self.match('PALABRA_RESERVADA', 'false'):
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

class ASTNode:
    def __init__(self, node_type, value=None, line=None, col=None, error=False):
        self.node_type = node_type
        self.value = value
        self.line = line
        self.col = col
        self.children = []
        self.is_error = error
    
    def add_child(self, child_node):
        self.children.append(child_node)
    
    def __repr__(self, level=0):
        prefix = "🚫 " if self.is_error else ""
        ret = "  " * level + f"{self.node_type}"
        if self.value:
            ret += f": {self.value}"
        if self.line and self.col:
            ret += f" [Línea: {self.line}, Col: {self.col}]"
        ret += "\n"
        
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret
    
class ASTViewer(QTreeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabel("Árbol Sintáctico Abstracto")
        self.setColumnCount(4)
        self.setHeaderLabels(["Nodo", "Valor", "Línea", "Columna"])
        
        # Ajustar el ancho de las columnas
        self.setColumnWidth(0, 150)  # Nodo
        self.setColumnWidth(1, 100)  # Valor
        self.setColumnWidth(2, 60)   # Línea
        self.setColumnWidth(3, 60)   # Columna
    
    def display_ast(self, ast_root):
        self.clear()
        if ast_root:
            self._add_tree_item(None, ast_root)
            self.expandAll()
    
    def _add_tree_item(self, parent_item, ast_node):
        # Crear texto para cada columna
        node_text = ast_node.node_type
        value_text = ast_node.value if ast_node.value else ""
        line_text = str(ast_node.line) if ast_node.line else ""
        col_text = str(ast_node.col) if ast_node.col else ""
        
        # Crear el ítem del árbol
        item = QTreeWidgetItem([node_text, value_text, line_text, col_text])
        
        if ast_node.is_error:
            # Fondo rojo para nodos con errores
            for i in range(4):  # Aplicar a todas las columnas
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
        
        # Configuración de los paneles y sus nombres
        self.tabs.addTab(self.output, "Salida")
        self.tabs.addTab(self.lexical_output, "Léxico")
        self.tabs.addTab(self.syntax_output, "Sintáctico")
        self.tabs.addTab(self.semantic_output, "Semántico")
        self.tabs.addTab(self.intermediate_code, "Código Intermedio")
        self.tabs.addTab(self.symbol_table, "Hash Table")
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
        
        # Icono de Ejecutar
        ejecutar_icon = self.style().standardIcon(QStyle.SP_MediaPlay)
        toolbar.addAction(ejecutar_icon, "Ejecutar", self.run_execution)
    
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
    
        if not hasattr(self, 'ast_viewer'):
            self.ast_viewer = ASTViewer()
            # Buscar el índice de la pestaña "Sintáctico"
            syntax_tab_index = -1
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i) == "Sintáctico":
                    syntax_tab_index = i
                    break
            
            if syntax_tab_index != -1:
                # Insertar después de la pestaña "Sintáctico"
                self.tabs.insertTab(syntax_tab_index + 1, self.ast_viewer, "Árbol AST")
            else:
                # Si no se encuentra, agregar al final
                self.tabs.addTab(self.ast_viewer, "Árbol AST")
        
        # Mostrar el AST
        self.ast_viewer.display_ast(ast)
        # Resaltar nodos con errores
        self.highlight_error_nodes(ast)
        # Cambiar a la pestaña del AST
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
        self.run_compiler("semantic")
        
    def run_intermediate(self):
        self.run_compiler("intermediate")
        
    def run_execution(self):
        self.run_compiler("execute")
        

# Función principal
# Esta función se encarga de iniciar la aplicación y mostrar la ventana principal.
if __name__ == '__main__':
    app = QApplication(sys.argv) # Crear la aplicación
    app.setStyle("Fusion") # Establecer el estilo de la aplicación
    ide = CompilerIDE() # Crear la ventana principal
    ide.show() # Mostrar la ventana
    sys.exit(app.exec_()) # Ejecutar la aplicación