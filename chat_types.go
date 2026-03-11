package llama

// ContentPart represents a single part of a multipart message.
// Used for vision/multimodal messages that contain both text and images.
//
// Example:
//
//	parts := []llama.ContentPart{
//	    {Type: "text", Text: "What's in this image?"},
//	    {Type: "image", Data: pngBytes},
//	}
type ContentPart struct {
	Type string // "text" or "image"
	Text string // Text content (for Type "text")
	Data []byte // Raw image bytes: jpg, png, bmp, gif (for Type "image")
}

// ChatMessage represents a message in a chat conversation.
//
// Common roles include "system", "user", "assistant", "tool", and "function".
// The role is not validated by this library - the model's chat template will
// handle role interpretation and any unknown roles.
//
// For text-only messages, use the Content field. For multipart messages
// (e.g., text + images), use Parts instead. When Parts is non-empty,
// it takes precedence over Content.
//
// Example (text-only):
//
//	messages := []llama.ChatMessage{
//	    {Role: "system", Content: "You are a helpful assistant."},
//	    {Role: "user", Content: "What is the capital of France?"},
//	}
//
// Example (vision):
//
//	messages := []llama.ChatMessage{
//	    {Role: "user", Parts: []llama.ContentPart{
//	        {Type: "text", Text: "Describe this image:"},
//	        {Type: "image", Data: imageBytes},
//	    }},
//	}
type ChatMessage struct {
	Role    string        // Message role (e.g., "system", "user", "assistant")
	Content string        // Simple text content (backward compatible)
	Parts   []ContentPart // Multipart content (takes precedence over Content when non-empty)
}

// ChatResponse represents the complete response from a chat completion.
//
// For standard models, only Content is populated. For reasoning models
// (like DeepSeek-R1), ReasoningContent may contain extracted thinking/
// reasoning tokens that were separated from the main response.
//
// Example:
//
//	response, err := model.Chat(ctx, messages, opts)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	fmt.Println("Response:", response.Content)
//	if response.ReasoningContent != "" {
//	    fmt.Println("Reasoning:", response.ReasoningContent)
//	}
type ChatResponse struct {
	Content          string // Regular response content
	ReasoningContent string // Extracted reasoning/thinking (if reasoning model)
	// Future fields: ToolCalls, FinishReason, Usage, etc.
}

// ChatDelta represents a streaming chunk from chat completion.
//
// During streaming, deltas arrive progressively. For standard models,
// only Content is populated with token(s). For reasoning models with
// extraction enabled, tokens may appear in either Content or
// ReasoningContent depending on whether they're inside reasoning tags.
//
// Example:
//
//	deltaCh, errCh := model.ChatStream(ctx, messages, opts)
//	for {
//	    select {
//	    case delta, ok := <-deltaCh:
//	        if !ok {
//	            return
//	        }
//	        if delta.Content != "" {
//	            fmt.Print(delta.Content)
//	        }
//	        if delta.ReasoningContent != "" {
//	            fmt.Print("[thinking: ", delta.ReasoningContent, "]")
//	        }
//	    case err := <-errCh:
//	        if err != nil {
//	            log.Fatal(err)
//	        }
//	    }
//	}
type ChatDelta struct {
	Content          string // Regular content token(s)
	ReasoningContent string // Reasoning token(s)
	// Future fields: ToolCalls, Role, FinishReason, etc.
}
