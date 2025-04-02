import Foundation

public enum LlamaCppSwiftError: Error {
    case decodeError
    case others(String)
}
