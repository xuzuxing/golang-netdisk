// Package deepseek implements DeepSeek API provider
// Author: Done-0
// Created: 2025-01-24
package deepseek

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"io"
	"log"
	"net/http"
	"time"

	"github.com/sashabaranov/go-openai"

	"github.com/Done-0/gin-scaffold/internal/ai/internal/provider"
	"github.com/Done-0/gin-scaffold/internal/ai/internal/selector"
	"github.com/Done-0/gin-scaffold/internal/ai/internal/types"
)

// Provider implements DeepSeek API provider
type Provider struct {
	ctx    *selector.ProviderContext
	client *openai.Client
}

// New creates a new DeepSeek provider instance from context
func New(ctx *selector.ProviderContext) (provider.Provider, error) {
	config := openai.DefaultConfig(ctx.APIKey)
	config.BaseURL = ctx.BaseURL
	config.HTTPClient = &http.Client{
		Timeout: time.Duration(ctx.Timeout) * time.Second,
		Transport: &http.Transport{
			MaxIdleConns:        100,
			MaxIdleConnsPerHost: 20,
			MaxConnsPerHost:     100,
			IdleConnTimeout:     90 * time.Second,
			DisableKeepAlives:   false,
		},
	}
	client := openai.NewClientWithConfig(config)

	return &Provider{
		ctx:    ctx,
		client: client,
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

	temperature := float32(p.ctx.Temperature)
	if req.Temperature > 0 {
		temperature = float32(req.Temperature)
	}

	messages := make([]openai.ChatCompletionMessage, len(req.Messages))
	for i, msg := range req.Messages {
		messages[i] = openai.ChatCompletionMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	request := openai.ChatCompletionRequest{
		Model:       model,
		Messages:    messages,
		MaxTokens:   maxTokens,
		Temperature: temperature,
		TopP:        float32(p.ctx.TopP),
	}

	var resp openai.ChatCompletionResponse
	var err error
	for attempt := 0; attempt <= p.ctx.MaxRetries; attempt++ {
		resp, err = p.client.CreateChatCompletion(ctx, request)
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

	responseID := resp.ID
	if responseID == "" {
		bytes := make([]byte, 16)
		rand.Read(bytes)
		responseID = "chatcmpl-" + hex.EncodeToString(bytes)
	}

	created := resp.Created
	if created == 0 {
		created = time.Now().Unix()
	}

	chatResp := &types.ChatResponse{
		ID:                responseID,
		Object:            resp.Object,
		Created:           created,
		Model:             resp.Model,
		SystemFingerprint: resp.SystemFingerprint,
		Provider:          p.ctx.ProviderName,
	}

	chatResp.Choices = make([]types.Choice, len(resp.Choices))
	for i, choice := range resp.Choices {
		chatResp.Choices[i] = types.Choice{
			Index: choice.Index,
			Message: types.Message{
				Role:             choice.Message.Role,
				Content:          choice.Message.Content,
				ReasoningContent: choice.Message.ReasoningContent,
			},
			FinishReason: string(choice.FinishReason),
		}
	}

	chatResp.Usage = types.Usage{
		PromptTokens:     resp.Usage.PromptTokens,
		CompletionTokens: resp.Usage.CompletionTokens,
		TotalTokens:      resp.Usage.TotalTokens,
	}
	if resp.Usage.CompletionTokensDetails != nil {
		chatResp.Usage.ReasoningTokens = resp.Usage.CompletionTokensDetails.ReasoningTokens
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

	temperature := float32(p.ctx.Temperature)
	if req.Temperature > 0 {
		temperature = float32(req.Temperature)
	}

	messages := make([]openai.ChatCompletionMessage, len(req.Messages))
	for i, msg := range req.Messages {
		messages[i] = openai.ChatCompletionMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	request := openai.ChatCompletionRequest{
		Model:         model,
		Messages:      messages,
		Stream:        true,
		MaxTokens:     maxTokens,
		Temperature:   temperature,
		TopP:          float32(p.ctx.TopP),
		StreamOptions: &openai.StreamOptions{IncludeUsage: true},
	}

	var stream *openai.ChatCompletionStream
	var err error
	for attempt := 0; attempt <= p.ctx.MaxRetries; attempt++ {
		stream, err = p.client.CreateChatCompletionStream(ctx, request)
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

	ch := make(chan *types.ChatStreamResponse)
	go func() {
		defer close(ch)
		defer stream.Close()

		for {
			response, err := stream.Recv()
			if err != nil {
				if err != io.EOF {
					log.Printf("Error receiving stream: %v", err)
				}
				break
			}

			responseID := response.ID
			if responseID == "" {
				bytes := make([]byte, 16)
				rand.Read(bytes)
				responseID = "chatcmpl-" + hex.EncodeToString(bytes)
			}

			created := response.Created
			if created == 0 {
				created = time.Now().Unix()
			}

			streamResp := &types.ChatStreamResponse{
				ID:                responseID,
				Object:            response.Object,
				Created:           created,
				Model:             response.Model,
				SystemFingerprint: response.SystemFingerprint,
				Provider:          p.ctx.ProviderName,
			}

			streamResp.Choices = make([]types.StreamChoice, len(response.Choices))
			for i, choice := range response.Choices {
				streamResp.Choices[i] = types.StreamChoice{
					Index: choice.Index,
					Delta: types.MessageDelta{
						Role:             choice.Delta.Role,
						Content:          choice.Delta.Content,
						ReasoningContent: choice.Delta.ReasoningContent,
					},
					FinishReason: string(choice.FinishReason),
				}
			}

			if response.Usage != nil {
				streamResp.Usage = &types.Usage{
					PromptTokens:     response.Usage.PromptTokens,
					CompletionTokens: response.Usage.CompletionTokens,
					TotalTokens:      response.Usage.TotalTokens,
				}
				if response.Usage.CompletionTokensDetails != nil {
					streamResp.Usage.ReasoningTokens = response.Usage.CompletionTokensDetails.ReasoningTokens
				}
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
