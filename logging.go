package llama

/*
#include "wrapper.h"
#include <stdlib.h>

// Forward declaration of Go log callback
extern void goLogCallback(int level, char* text, void* user_data);

static inline llama_wrapper_log_callback get_go_log_callback() {
	return (llama_wrapper_log_callback)goLogCallback;
}
*/
import "C"
import (
	"fmt"
	"strings"
	"unsafe"
)

// LogLevel represents the severity of a log message from llama.cpp.
type LogLevel int

const (
	LogLevelNone  LogLevel = 0
	LogLevelDebug LogLevel = 1
	LogLevelInfo  LogLevel = 2
	LogLevelWarn  LogLevel = 3
	LogLevelError LogLevel = 4
)

// String returns the human-readable name of the log level.
func (l LogLevel) String() string {
	switch l {
	case LogLevelDebug:
		return "debug"
	case LogLevelInfo:
		return "info"
	case LogLevelWarn:
		return "warn"
	case LogLevelError:
		return "error"
	default:
		return "none"
	}
}

// LogCallback is a function that receives log messages from llama.cpp.
// The message may contain trailing newlines.
type LogCallback func(level LogLevel, message string)

// globalLogCallback holds the current Go log callback.
var globalLogCallback LogCallback

// SetLogCallback installs a custom log callback that receives all llama.cpp
// log messages (filtered by the current LLAMA_LOG level). Pass nil to restore
// the default behaviour of writing to stderr.
//
// Example:
//
//	llama.SetLogCallback(func(level llama.LogLevel, message string) {
//	    fmt.Printf("[%s] %s", level, message)
//	})
func SetLogCallback(cb LogCallback) {
	globalLogCallback = cb
	if cb != nil {
		C.llama_wrapper_set_log_callback(C.get_go_log_callback(), nil)
	} else {
		C.llama_wrapper_set_log_callback(nil, nil)
	}
}

//export goLogCallback
func goLogCallback(level C.int, text *C.char, _ unsafe.Pointer) {
	if globalLogCallback != nil {
		msg := strings.TrimRight(C.GoString(text), "\n")
		if msg != "" {
			globalLogCallback(LogLevel(level), msg)
		}
	}
}

// logInfo sends a formatted info message through the global log callback.
func logInfo(format string, args ...any) {
	if globalLogCallback != nil {
		globalLogCallback(LogLevelInfo, fmt.Sprintf(format, args...))
	}
}
