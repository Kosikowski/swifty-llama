import Foundation

/// Grammar constraint for structured output generation
///
/// Grammar constraints enforce specific output formats during text generation.
/// They work by filtering the vocabulary at each step to only allow tokens
/// that are valid according to the grammar rules.
///
/// Example usage:
/// ```swift
/// // Use a preset grammar
/// let params = GenerationParams(grammar: .json)
///
/// // Use a custom grammar
/// let customGrammar = GrammarConstraint(
///     rules: """
///     root ::= object
///     object ::= "{" ws members? "}" ws
///     members ::= pair ("," ws pair)*
///     pair ::= string ":" ws value
///     ...
///     """,
///     rootSymbol: "root"
/// )
/// let params = GenerationParams(grammar: customGrammar)
/// ```
public struct GrammarConstraint: Hashable, Sendable {
    /// The grammar rules in GBNF (GGML BNF) format
    public let rules: String

    /// The root symbol name to start parsing from
    public let rootSymbol: String

    /// Initialize with custom grammar rules
    /// - Parameters:
    ///   - rules: Grammar rules in GBNF format
    ///   - rootSymbol: The root symbol name (default: "root")
    public init(rules: String, rootSymbol: String = "root") {
        self.rules = rules
        self.rootSymbol = rootSymbol
    }
}

// MARK: - Common Grammar Presets

public extension GrammarConstraint {
    /// JSON grammar for generating valid JSON output
    ///
    /// Supports: objects, arrays, strings, numbers, booleans, null
    static let json = GrammarConstraint(
        rules: """
        root   ::= object
        value  ::= object | array | string | number | ("true" | "false" | "null") ws

        object ::=
          "{" ws (
                    string ":" ws value
            ("," ws string ":" ws value)*
          )? "}" ws

        array  ::=
          "[" ws (
                    value
            ("," ws value)*
          )? "]" ws

        string ::=
          "\\"" (
            [^"\\\\] |
            "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
          )* "\\"" ws

        number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

        ws ::= ([ \\t\\n] ws)?
        """,
        rootSymbol: "root"
    )

    /// JSON array grammar for generating JSON arrays
    static let jsonArray = GrammarConstraint(
        rules: """
        root   ::= array
        value  ::= object | array | string | number | ("true" | "false" | "null") ws

        object ::=
          "{" ws (
                    string ":" ws value
            ("," ws string ":" ws value)*
          )? "}" ws

        array  ::=
          "[" ws (
                    value
            ("," ws value)*
          )? "]" ws

        string ::=
          "\\"" (
            [^"\\\\] |
            "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
          )* "\\"" ws

        number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

        ws ::= ([ \\t\\n] ws)?
        """,
        rootSymbol: "root"
    )

    /// List grammar for generating numbered or bulleted lists
    static let list = GrammarConstraint(
        rules: """
        root        ::= item+
        item        ::= bullet text newline
        bullet      ::= ("- " | "* " | [0-9]+ ". ")
        text        ::= [^\\n]+
        newline     ::= "\\n"
        """,
        rootSymbol: "root"
    )

    /// Boolean grammar - outputs only "true" or "false"
    static let boolean = GrammarConstraint(
        rules: """
        root ::= ("true" | "false")
        """,
        rootSymbol: "root"
    )

    /// Yes/No grammar - outputs only "yes" or "no" (case insensitive start)
    static let yesNo = GrammarConstraint(
        rules: """
        root ::= (("Y" | "y") "es" | ("N" | "n") "o")
        """,
        rootSymbol: "root"
    )

    /// Integer grammar - outputs only integer numbers
    static let integer = GrammarConstraint(
        rules: """
        root ::= "-"? ([0-9] | [1-9] [0-9]*)
        """,
        rootSymbol: "root"
    )

    /// Decimal number grammar
    static let decimal = GrammarConstraint(
        rules: """
        root ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)?
        """,
        rootSymbol: "root"
    )
}
