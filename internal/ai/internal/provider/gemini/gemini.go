// Package gemini implements Google Gemini API provider
// Author: Done-0
// Created: 2025-01-24
package gemini

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"time"

	"google.golang.org/genai"

	"github.com/Done-0/gin-scaffold/internal/ai/internal/provider"
	"github.com/Done-0/gin-scaffold/internal/ai/internal/selector"
	"github.com/Done-0/gin-scaffold/internal/ai/internal/types"
)

// Provider implements Google Gemini API provider
type Provider struct {
	ctx *selector.ProviderContext
}

// New creates a new Gemini provider instance from context
func New(ctx *selector.ProviderContext) (provider.Provider, error) {
	return &Provider{
		ctx: ctx,
	}, nil
}

// Chat sends a chat completion request
func (p *Provider) Chat(ctx context.Context, req *types.ChatRequest) (*types.ChatResponse, error) {
	model := req.Model
	if model == "" {
		model = p.ctx.Model
	}

	clientConfig := &genai.ClientConfig{
		APIKey:  p.ctx.APIKey,
		Backend: genai.BackendGeminiAPI,
	}

	client, err := genai.NewClient(ctx, clientConfig)
	if err != nil {
		return nil, err
	}

	maxTokens := p.ctx.MaxTokens
	if req.MaxTokens > 0 {
		maxTokens = req.MaxTokens
	}

	temperature := float32(p.ctx.Temperature)
	if req.Temperature > 0 {
		temperature = float32(req.Temperature)
	}

	contents := make([]*genai.Content, len(req.Messages))
	for i, msg := range req.Messages {
		role := genai.RoleUser
		if msg.Role == "assistant" {
			role = genai.RoleModel
		}
		contents[i] = &genai.Content{
			Parts: []*genai.Part{{Text: msg.Content}},
			Role:  role,
		}
	}

	var resp *genai.GenerateContentResponse
	for attempt := 0; attempt <= p.ctx.MaxRetries; attempt++ {
		resp, err = client.Models.GenerateContent(
			ctx,
			model,
			contents,
			&genai.GenerateContentConfig{
				Temperature:     &temperature,
				MaxOutputTokens: int32(maxTokens),
				TopP:            func() *float32 { v := float32(p.ctx.TopP); return &v }(),
				TopK:            func() *float32 { v := float32(p.ctx.TopK); return &v }(),
				ThinkingConfig: &genai.ThinkingConfig{
					IncludeThoughts: true,
				},
			},
		)
		if err == nil {
			break
		}
		if attempt < p.ctx.MaxRetries {
			time.Sleep(time.Duration(attempt+1) * time.Second)
		}
	}
	if err != nil {
		return nil, err
	}

	responseID := resp.ResponseID
	if responseID == "" {
		bytes := make([]byte, 16)
		rand.Read(bytes)
		responseID = "chatcmpl-" + hex.EncodeToString(bytes)
	}

	var created int64
	if resp.CreateTime.IsZero() {
		created = time.Now().Unix()
	} else {
		created = resp.CreateTime.Unix()
	}

	chatResp := &types.ChatResponse{
		ID:                responseID,
		Object:            "chat.completion",
		Created:           created,
		Model:             model,
		SystemFingerprint: "",
		Provider:          p.ctx.ProviderName,
	}

	chatResp.Choices = make([]types.Choice, len(resp.Candidates))
	for i, candidate := range resp.Candidates {
		var content, reasoningContent string
		if candidate.Content != nil {
			for _, part := range candidate.Content.Parts {
				if len(part.Text) == 0 {
					continue
				}
				if part.Thought {
					reasoningContent += part.Text
				} else {
					content += part.Text
				}
			}
		}
		chatResp.Choices[i] = types.Choice{
			Index: i,
			Message: types.Message{
				Role:             "assistant",
				Content:          content,
				ReasoningContent: reasoningContent,
			},
			FinishReason: string(candidate.FinishReason),
		}
	}

	if resp.UsageMetadata != nil {
		chatResp.Usage = types.Usage{
			PromptTokens:     int(resp.UsageMetadata.PromptTokenCount),
			CompletionTokens: int(resp.UsageMetadata.CandidatesTokenCount),
			TotalTokens:      int(resp.UsageMetadata.TotalTokenCount),
			ReasoningTokens:  int(resp.UsageMetadata.ThoughtsTokenCount),
		}
	}

	return chatResp, nil
}

// ChatStream sends a streaming chat completion request
func (p *Provider) ChatStream(ctx context.Context, req *types.ChatRequest) (<-chan *types.ChatStreamResponse, error) {
	model := req.Model
	if model == "" {
		model = p.ctx.Model
	}

	clientConfig := &genai.ClientConfig{
		APIKey:  p.ctx.APIKey,
		Backend: genai.BackendGeminiAPI,
	}

	maxTokens := p.ctx.MaxTokens
	if req.MaxTokens > 0 {
		maxTokens = req.MaxTokens
	}

	temperature := float32(p.ctx.Temperature)
	if req.Temperature > 0 {
		temperature = float32(req.Temperature)
	}

	contents := make([]*genai.Content, len(req.Messages))
	for i, msg := range req.Messages {
		role := genai.RoleUser
		if msg.Role == "assistant" {
			role = genai.RoleModel
		}
		contents[i] = &genai.Content{
			Parts: []*genai.Part{{Text: msg.Content}},
			Role:  role,
		}
	}

	var client *genai.Client
	var err error
	for attempt := 0; attempt <= p.ctx.MaxRetries; attempt++ {
		client, err = genai.NewClient(ctx, clientConfig)
		if err == nil {
			break
		}
		if attempt < p.ctx.MaxRetries {
			time.Sleep(time.Duration(attempt+1) * time.Second)
		}
	}
	if err != nil {
		return nil, err
	}

	stream := client.Models.GenerateContentStream(
		ctx,
		model,
		contents,
		&genai.GenerateContentConfig{
			Temperature:     &temperature,
			MaxOutputTokens: int32(maxTokens),
			TopP:            func() *float32 { v := float32(p.ctx.TopP); return &v }(),
			TopK:            func() *float32 { v := float32(p.ctx.TopK); return &v }(),
			ThinkingConfig: &genai.ThinkingConfig{
				IncludeThoughts: true,
			},
		},
	)

	ch := make(chan *types.ChatStreamResponse)
	go func() {
		defer close(ch)

		if stream == nil {
			return
		}

		for chunk := range stream {
			if chunk == nil {
				continue
			}

			if len(chunk.Candidates) == 0 {
				continue
			}

			responseID := chunk.ResponseID
			if responseID == "" {
				bytes := make([]byte, 16)
				rand.Read(bytes)
				responseID = "chatcmpl-" + hex.EncodeToString(bytes)
			}

			var created int64
			if chunk.CreateTime.IsZero() {
				created = time.Now().Unix()
			} else {
				created = chunk.CreateTime.Unix()
			}

			streamResp := &types.ChatStreamResponse{
				ID:                responseID,
				Object:            "chat.completion.chunk",
				Created:           created,
				Model:             model,
				SystemFingerprint: "",
				Provider:          p.ctx.ProviderName,
			}

			streamResp.Choices = make([]types.StreamChoice, 0, len(chunk.Candidates))
			var hasFinished bool
			for i, candidate := range chunk.Candidates {
				if candidate.Content == nil || len(candidate.Content.Parts) == 0 {
					continue
				}

				var normalContent, thinkingContent string
				for _, part := range candidate.Content.Parts {
					if part.Text == "" {
						continue
					}
					if part.Thought {
						thinkingContent += part.Text
					} else {
						normalContent += part.Text
					}
				}

				streamResp.Choices = append(streamResp.Choices, types.StreamChoice{
					Index: i,
					Delta: types.MessageDelta{
						Role:             "assistant",
						Content:          normalContent,
						ReasoningContent: thinkingContent,
					},
					FinishReason: string(candidate.FinishReason),
				})

				if candidate.FinishReason == genai.FinishReasonStop || candidate.FinishReason == genai.FinishReasonMaxTokens {
					hasFinished = true
				}
			}

			if len(streamResp.Choices) == 0 {
				continue
			}

			if chunk.UsageMetadata != nil {
				streamResp.Usage = &types.Usage{
					PromptTokens:     int(chunk.UsageMetadata.PromptTokenCount),
					CompletionTokens: int(chunk.UsageMetadata.CandidatesTokenCount),
					TotalTokens:      int(chunk.UsageMetadata.TotalTokenCount),
					ReasoningTokens:  int(chunk.UsageMetadata.ThoughtsTokenCount),
				}
			}

			select {
			case ch <- streamResp:
			case <-ctx.Done():
				return
			}

			if hasFinished {
				break
			}
		}
	}()

	return ch, nil
}
