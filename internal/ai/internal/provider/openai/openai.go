// Package openai implements OpenAI-compatible API provider
// Author: Done-0
// Created: 2025-01-24
package openai

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/Done-0/gin-scaffold/internal/ai/internal/provider"
	"github.com/Done-0/gin-scaffold/internal/ai/internal/selector"
	"github.com/Done-0/gin-scaffold/internal/ai/internal/types"
)

// Provider implements OpenAI-compatible API provider
type Provider struct {
	ctx        *selector.ProviderContext
	httpClient *http.Client
}

// New creates a new OpenAI provider instance from context
func New(ctx *selector.ProviderContext) (provider.Provider, error) {
	return &Provider{
		ctx: ctx,
		httpClient: &http.Client{
			Timeout: time.Duration(ctx.Timeout) * time.Second,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 20,
				MaxConnsPerHost:     100,
				IdleConnTimeout:     90 * time.Second,
				DisableKeepAlives:   false,
			},
		},
	}, nil
}

// Chat sends a chat completion request
func (p *Provider) Chat(ctx context.Context, req *types.ChatRequest) (*types.ChatResponse, error) {
	model := req.Model
	if model == "" {
		model = p.ctx.Model
	}

	maxTokens := p.ctx.MaxTokens
	if req.MaxTokens > 0 {
		maxTokens = req.MaxTokens
	}

	temperature := p.ctx.Temperature
	if req.Temperature > 0 {
		temperature = req.Temperature
	}

	// Build request body matching TypeScript version
	messages := make([]map[string]any, len(req.Messages))
	for i, msg := range req.Messages {
		messages[i] = map[string]any{
			"role":    msg.Role,
			"content": msg.Content,
		}
	}

	requestBody := map[string]any{
		"model":       model,
		"messages":    messages,
		"max_tokens":  maxTokens,
		"temperature": temperature,
		"top_p":       p.ctx.TopP,
		"extra_body": map[string]any{
			"google": map[string]any{
				"thinking_config": map[string]any{
					"include_thoughts": true,
				},
			},
		},
	}

	bodyBytes, err := json.Marshal(requestBody)
	if err != nil {
		return nil, err
	}

	var resp *http.Response
	var lastError error

	for attempt := 0; attempt <= p.ctx.MaxRetries; attempt++ {
		httpReq, err := http.NewRequestWithContext(ctx, "POST", p.ctx.BaseURL+"/chat/completions", bytes.NewReader(bodyBytes))
		if err != nil {
			return nil, err
		}

		httpReq.Header.Set("Authorization", "Bearer "+p.ctx.APIKey)
		httpReq.Header.Set("Content-Type", "application/json")

		resp, err = p.httpClient.Do(httpReq)
		if err == nil && resp.StatusCode == http.StatusOK {
			break
		}

		lastError = err
		if resp != nil {
			resp.Body.Close()
		}

		if attempt < p.ctx.MaxRetries {
			time.Sleep(time.Duration(attempt+1) * time.Second)
		}
	}

	if resp == nil {
		if lastError != nil {
			return nil, lastError
		}
		return nil, fmt.Errorf("HTTP request failed: no response")
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, fmt.Errorf("HTTP error: %d", resp.StatusCode)
	}
	defer resp.Body.Close()

	var result struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		Created int64  `json:"created"`
		Model   string `json:"model"`
		Choices []struct {
			Index   int `json:"index"`
			Message struct {
				Role             string `json:"role"`
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens            int `json:"prompt_tokens"`
			CompletionTokens        int `json:"completion_tokens"`
			TotalTokens             int `json:"total_tokens"`
			CompletionTokensDetails *struct {
				ReasoningTokens int `json:"reasoning_tokens"`
			} `json:"completion_tokens_details"`
		} `json:"usage"`
		SystemFingerprint string `json:"system_fingerprint"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	responseID := result.ID
	if responseID == "" {
		bytes := make([]byte, 16)
		rand.Read(bytes)
		responseID = "chatcmpl-" + hex.EncodeToString(bytes)
	}

	created := result.Created
	if created == 0 {
		created = time.Now().Unix()
	}

	chatResp := &types.ChatResponse{
		ID:                responseID,
		Object:            result.Object,
		Created:           created,
		Model:             result.Model,
		SystemFingerprint: result.SystemFingerprint,
		Provider:          p.ctx.ProviderName,
	}

	chatResp.Choices = make([]types.Choice, len(result.Choices))
	for i, choice := range result.Choices {
		chatResp.Choices[i] = types.Choice{
			Index: choice.Index,
			Message: types.Message{
				Role:             choice.Message.Role,
				Content:          choice.Message.Content,
				ReasoningContent: choice.Message.ReasoningContent,
			},
			FinishReason: choice.FinishReason,
		}
	}

	chatResp.Usage = types.Usage{
		PromptTokens:     result.Usage.PromptTokens,
		CompletionTokens: result.Usage.CompletionTokens,
		TotalTokens:      result.Usage.TotalTokens,
	}

	if result.Usage.CompletionTokensDetails != nil {
		chatResp.Usage.ReasoningTokens = result.Usage.CompletionTokensDetails.ReasoningTokens
	}

	return chatResp, nil
}

// ChatStream sends a streaming chat completion request
func (p *Provider) ChatStream(ctx context.Context, req *types.ChatRequest) (<-chan *types.ChatStreamResponse, error) {
	model := req.Model
	if model == "" {
		model = p.ctx.Model
	}

	maxTokens := p.ctx.MaxTokens
	if req.MaxTokens > 0 {
		maxTokens = req.MaxTokens
	}

	temperature := p.ctx.Temperature
	if req.Temperature > 0 {
		temperature = req.Temperature
	}

	// Build request body matching TypeScript version
	messages := make([]map[string]any, len(req.Messages))
	for i, msg := range req.Messages {
		messages[i] = map[string]any{
			"role":    msg.Role,
			"content": msg.Content,
		}
	}

	requestBody := map[string]any{
		"model":       model,
		"messages":    messages,
		"max_tokens":  maxTokens,
		"temperature": temperature,
		"top_p":       p.ctx.TopP,
		"stream":      true,
		"stream_options": map[string]any{
			"include_usage": true,
		},
		"extra_body": map[string]any{
			"google": map[string]any{
				"thinking_config": map[string]any{
					"include_thoughts": true,
				},
			},
		},
		"tools": []map[string]any{
			{
				"type": "function",
				"function": map[string]any{
					"name":        "_dummy",
					"description": "Dummy function for enabling thinking",
					"parameters": map[string]any{
						"type":       "object",
						"properties": map[string]any{},
					},
				},
			},
		},
		"tool_choice": "auto",
	}

	bodyBytes, err := json.Marshal(requestBody)
	if err != nil {
		return nil, err
	}

	var resp *http.Response
	var lastError error

	for attempt := 0; attempt <= p.ctx.MaxRetries; attempt++ {
		httpReq, err := http.NewRequestWithContext(ctx, "POST", p.ctx.BaseURL+"/chat/completions", bytes.NewReader(bodyBytes))
		if err != nil {
			return nil, err
		}

		httpReq.Header.Set("Authorization", "Bearer "+p.ctx.APIKey)
		httpReq.Header.Set("Content-Type", "application/json")

		resp, err = p.httpClient.Do(httpReq)
		if err == nil && resp.StatusCode == http.StatusOK {
			break
		}

		lastError = err
		if resp != nil {
			resp.Body.Close()
		}

		if attempt < p.ctx.MaxRetries {
			time.Sleep(time.Duration(attempt+1) * time.Second)
		}
	}

	if resp == nil {
		if lastError != nil {
			return nil, lastError
		}
		return nil, fmt.Errorf("HTTP request failed: no response")
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("HTTP error: %d", resp.StatusCode)
	}

	ch := make(chan *types.ChatStreamResponse)
	go func() {
		defer close(ch)
		defer resp.Body.Close()

		reader := bufio.NewReader(resp.Body)
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				if err != io.EOF {
					log.Printf("Error reading stream: %v", err)
				}
				break
			}

			line = strings.TrimSpace(line)
			if !strings.HasPrefix(line, "data: ") {
				continue
			}

			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				break
			}

			var chunk struct {
				ID      string `json:"id"`
				Object  string `json:"object"`
				Created int64  `json:"created"`
				Model   string `json:"model"`
				Choices []struct {
					Index int `json:"index"`
					Delta struct {
						Role             string `json:"role"`
						Content          string `json:"content"`
						ReasoningContent string `json:"reasoning_content"`
					} `json:"delta"`
					FinishReason string `json:"finish_reason"`
				} `json:"choices"`
				SystemFingerprint string `json:"system_fingerprint"`
				Usage             *struct {
					PromptTokens            int `json:"prompt_tokens"`
					CompletionTokens        int `json:"completion_tokens"`
					TotalTokens             int `json:"total_tokens"`
					CompletionTokensDetails *struct {
						ReasoningTokens int `json:"reasoning_tokens"`
					} `json:"completion_tokens_details"`
					ReasoningTokens int `json:"reasoning_tokens"`
				} `json:"usage"`
			}

			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				log.Printf("Failed to parse stream chunk: %v", err)
				continue
			}

			responseID := chunk.ID
			if responseID == "" {
				bytes := make([]byte, 16)
				rand.Read(bytes)
				responseID = "chatcmpl-" + hex.EncodeToString(bytes)
			}

			created := chunk.Created
			if created == 0 {
				created = time.Now().Unix()
			}

			streamResp := &types.ChatStreamResponse{
				ID:                responseID,
				Object:            chunk.Object,
				Created:           created,
				Model:             chunk.Model,
				SystemFingerprint: chunk.SystemFingerprint,
				Provider:          p.ctx.ProviderName,
			}

			streamResp.Choices = make([]types.StreamChoice, len(chunk.Choices))
			for i, choice := range chunk.Choices {
				streamResp.Choices[i] = types.StreamChoice{
					Index: choice.Index,
					Delta: types.MessageDelta{
						Role:             choice.Delta.Role,
						Content:          choice.Delta.Content,
						ReasoningContent: choice.Delta.ReasoningContent,
					},
					FinishReason: choice.FinishReason,
				}
			}

			if chunk.Usage != nil {
				usage := &types.Usage{
					PromptTokens:     chunk.Usage.PromptTokens,
					CompletionTokens: chunk.Usage.CompletionTokens,
					TotalTokens:      chunk.Usage.TotalTokens,
				}

				// Extract reasoning tokens from completion_tokens_details or reasoning_tokens
				if chunk.Usage.CompletionTokensDetails != nil {
					usage.ReasoningTokens = chunk.Usage.CompletionTokensDetails.ReasoningTokens
				} else if chunk.Usage.ReasoningTokens > 0 {
					usage.ReasoningTokens = chunk.Usage.ReasoningTokens
				}

				streamResp.Usage = usage
			}

			select {
			case ch <- streamResp:
			case <-ctx.Done():
				return
			}
		}
	}()

	return ch, nil
}
