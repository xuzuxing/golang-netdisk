// Package provider defines AI provider interface
// Author: Done-0
// Created: 2025-08-31
package provider

import (
	"context"

	"github.com/Done-0/gin-scaffold/internal/ai/internal/types"
)

// Provider is the interface for AI providers
type Provider interface {
	Chat(ctx context.Context, req *types.ChatRequest) (*types.ChatResponse, error)
	ChatStream(ctx context.Context, req *types.ChatRequest) (<-chan *types.ChatStreamResponse, error)
}
